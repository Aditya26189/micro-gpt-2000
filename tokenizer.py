"""BPE tokenizer (V=2048) with byte-level fallback.

Trained on the provided corpus only.  Lossless: decode(encode(text)) == text.
Uses word-frequency pre-tokenisation for fast training (~5-10 min on 7 MB).

Interface (required by evaluate.py):
    load()           -> tokenizer with .encode(), .decode(), .vocab_size
    .encode(str)     -> list[int]
    .decode(list)    -> str
"""
import collections
import json
import os
import re


# ---------------------------------------------------------------------------
# Byte-level fallback (kept for reference / emergency)
# ---------------------------------------------------------------------------
class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


# ---------------------------------------------------------------------------
# BPE tokenizer
# ---------------------------------------------------------------------------
class BPETokenizer:
    """Byte-Pair Encoding with a base-256 byte vocabulary.

    Training uses pre-tokenisation (`\\S+|\\s+` on raw bytes) and
    word-frequency counting so that the merge loop operates on unique
    chunks rather than the full 7 M-byte corpus.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new_id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size: int = 256

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of *pair* in *ids* with *new_id*."""
        out: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    _SPLIT_RE = re.compile(rb"\S+|\s+")

    # -- training ---------------------------------------------------------
    def train(self, text: str, target_vocab_size: int = 2048) -> None:
        raw = text.encode("utf-8")
        chunks = self._SPLIT_RE.findall(raw)

        # word-frequency table: {chunk_bytes: frequency}
        freq: dict[bytes, int] = collections.Counter(chunks)

        # id-sequence for each unique chunk
        word_ids: dict[bytes, list[int]] = {ch: list(ch) for ch in freq}

        # Initial pair counts and inverted index
        pair_counts: dict[tuple[int, int], int] = collections.defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[bytes]] = collections.defaultdict(set)
        for ch, f in freq.items():
            ids = word_ids[ch]
            for j in range(len(ids) - 1):
                pair = (ids[j], ids[j + 1])
                pair_counts[pair] += f
                pair_to_words[pair].add(ch)

        num_merges = target_vocab_size - 256
        for i in range(num_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] <= 0:
                break
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            affected_words = list(pair_to_words[best])
            if best in pair_counts:
                del pair_counts[best]
            if best in pair_to_words:
                del pair_to_words[best]

            for ch in affected_words:
                old_ids = word_ids[ch]
                f = freq[ch]
                # Subtract old pairs from counts and index
                for j in range(len(old_ids) - 1):
                    pair = (old_ids[j], old_ids[j + 1])
                    if pair in pair_counts:
                        pair_counts[pair] -= f
                        if pair_counts[pair] <= 0:
                            del pair_counts[pair]
                    if pair in pair_to_words:
                        pair_to_words[pair].discard(ch)

                # Apply merge
                new_ids = self._merge(old_ids, best, new_id)
                word_ids[ch] = new_ids

                # Add new pairs to counts and index
                for j in range(len(new_ids) - 1):
                    pair = (new_ids[j], new_ids[j + 1])
                    pair_counts[pair] += f
                    pair_to_words[pair].add(ch)

            if (i + 1) % 200 == 0 or (i + 1) == num_merges:
                print(f"  BPE merge {i + 1}/{num_merges}", flush=True)

        self.vocab_size = 256 + len(self.merges)
        print(f"  BPE training complete: {self.vocab_size} tokens ({len(self.merges)} merges)", flush=True)

    # -- encode / decode --------------------------------------------------
    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        chunks = self._SPLIT_RE.findall(raw)
        all_ids: list[int] = []
        for chunk in chunks:
            ids = list(chunk)
            # Repeatedly apply the lowest-index applicable merge
            while len(ids) >= 2:
                best_pair = None
                best_idx = float("inf")
                for j in range(len(ids) - 1):
                    pair = (ids[j], ids[j + 1])
                    midx = self.merges.get(pair)
                    if midx is not None and midx < best_idx:
                        best_idx = midx
                        best_pair = pair
                if best_pair is None:
                    break
                ids = self._merge(ids, best_pair, self.merges[best_pair])
            all_ids.extend(ids)
        return all_ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path) as f:
            data = json.load(f)
        tok.vocab_size = data["vocab_size"]
        # Rebuild merges & vocab in merge-index order
        items = sorted(data["merges"].items(), key=lambda kv: kv[1])
        for key_str, c in items:
            a, b = (int(x) for x in key_str.split(","))
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        return tok


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load(path=None):
    """Return the tokenizer used by evaluate.py.  Called with NO arguments."""
    base = os.path.dirname(os.path.abspath(__file__))
    tok_path = path or os.path.join(base, "bpe_tokenizer.json")
    if os.path.exists(tok_path):
        return BPETokenizer.from_file(tok_path)
    raise FileNotFoundError(
        f"BPE tokenizer not found at {tok_path}.  "
        "Run  python train.py --data ../data/train_corpus.txt  to auto-train it."
    )
