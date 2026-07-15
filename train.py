"""Fast trainer — target ≤10 min for 2000 steps on laptop CPU.

Speed choices:
  • Corpus token cache (train_ids.pt) — skips ~60–90 s re-encode each run
  • 6-layer GeLU stack (no recurrence / SwiGLU / EMA)
  • block_size=256, batch=32 defaults
  • Cosine LR schedule (cheaper than WSD bookkeeping, same class of fix)

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import os
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod
from tokenizer import BPETokenizer

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i : i + block] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def cosine_lr(step, total, warmup, peak, floor_ratio=0.1):
    if step <= 0:
        return 0.0
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = peak * floor_ratio
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def weight_decay_from_epoch_ratio(ratio: float) -> float:
    if ratio < 2.0:
        return 0.02
    if ratio < 5.0:
        return 0.04
    return 0.06


def load_corpus_ids(text: str, tok, cache_path: str) -> torch.Tensor:
    """Load token ids from cache or encode once and save."""
    if os.path.exists(cache_path):
        print(f"  loading cached ids: {cache_path}")
        return torch.load(cache_path, weights_only=True)
    print("  encoding corpus (one-time, then cached)...")
    t0 = time.time()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save(ids, cache_path)
    print(f"  encoded {len(ids):,} tokens in {time.time() - t0:.1f}s -> {cache_path}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--reencode", action="store_true",
                    help="Ignore train_ids.pt cache and re-encode corpus")
    args = ap.parse_args()

    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    device = "cpu"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    text = open(args.data, encoding="utf-8").read()
    n_bytes = len(text.encode("utf-8"))

    # --- BPE tokenizer (train once) ----------------------------------------
    tok_path = os.path.join(base_dir, "bpe_tokenizer.json")
    if not os.path.exists(tok_path):
        print("--- Training BPE tokenizer (one-time) ---")
        bpe = BPETokenizer()
        bpe.train(text, target_vocab_size=2048)
        bpe.save(tok_path)
        assert bpe.decode(bpe.encode(text[:10_000])) == text[:10_000]
        print("  [OK] BPE trained")

    tok = tokenizer_mod.load()

    # --- lossless checks (dev full, train sample) --------------------------
    dev_path = os.path.join(os.path.dirname(args.data), "dev_eval.txt")
    if os.path.exists(dev_path):
        dev_text = open(dev_path, encoding="utf-8").read()
        assert tok.decode(tok.encode(dev_text)) == dev_text, \
            "FATAL: tokenizer not lossless on dev_eval.txt"
        print("  [OK] dev_eval.txt round-trip")

    ids_cache = os.path.join(base_dir, "train_ids.pt")
    if args.reencode and os.path.exists(ids_cache):
        os.remove(ids_cache)
    ids = load_corpus_ids(text, tok, ids_cache)

    bpe_tokens = len(ids)
    bytes_per_token = n_bytes / bpe_tokens
    tokens_per_step = args.batch * Config.block_size
    epoch_ratio = (args.steps * tokens_per_step) / bpe_tokens
    wd = weight_decay_from_epoch_ratio(epoch_ratio)

    print(f"corpus: {n_bytes:,} bytes -> {bpe_tokens:,} tokens "
          f"(vocab {tok.vocab_size}, {bytes_per_token:.2f} bytes/token)")
    print(f"  epoch_ratio={epoch_ratio:.2f}  (batch={args.batch}, "
          f"block={Config.block_size}) -> weight_decay={wd}")

    # --- model -------------------------------------------------------------
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (cap: {MAX_PARAMS:,})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params  (have {n:,})"

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=wd,
    )

    model.train()
    t0 = time.time()
    losses = []

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if step == 50:
            ms = (time.time() - t0) / step * 1000
            eta_min = ms * args.steps / 60000
            print(f"  >> ETA full run: {eta_min:.1f} min  ({ms:.0f} ms/step)")

        if step % args.log_every == 0 or step == 1:
            recent = losses[-args.log_every:]
            avg = sum(recent) / len(recent)
            ms = (time.time() - t0) / step * 1000
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.6f}  ({ms:.0f} ms/step)")

    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                k: getattr(cfg, k)
                for k in dir(cfg)
                if not k.startswith("_") and not callable(getattr(cfg, k))
            },
            "steps": args.steps,
            "train_loss_curve": losses,
            "epoch_ratio": epoch_ratio,
            "weight_decay": wd,
        },
        args.out,
    )
    elapsed = time.time() - t0
    print(f"saved {args.out}  ({elapsed:.0f}s = {elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
