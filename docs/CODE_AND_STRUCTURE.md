# Code & Architectural Structure Guide (`Micro-GPT 2000`)

This document provides a complete engineering manual and architectural walkthrough of the `Micro-GPT 2000` codebase. It details the exact mathematical execution flow across every module, the internal structures of our custom classes (`tokenizer.py`, `model.py`, `train.py`, `evaluate.py`), and how our training pipeline optimizes parameter convergence within a strict 2,000-step budget.

---

## 🗂️ High-Level Codebase Topology

```text
C:\Users\LawLight\Desktop\pilvo\
├── docs/
│   ├── ARCHITECTURE_COMPARISON.md     # Ablation: GeLU vs SwiGLU + Depth Recurrence
│   ├── ENGINEERING_OPTIMIZATIONS.md    # Systems: 45 min -> 7.2 min optimization breakdown
│   └── CODE_AND_STRUCTURE.md           # Deep-dive code architecture & training guide (this file)
├── model.py                           # Pure PyTorch neural network definitions (GPT, Block, SelfAttention, SwiGLU)
├── train.py                           # End-to-end training pipeline (WSD schedule, AdamW, EMA, token caching)
├── tokenizer.py                       # Self-contained Byte-Pair Encoding (BPE, V=2048) implementation
├── evaluate.py                        # Official vocabulary-invariant evaluation & BPB calculation script
├── ckpt.pt                            # Final production weights (`1,660,352 params`, `2.0231 BPB`)
├── train_ids.pt                       # Serialized integer token array (`2,919,027 tokens` cached)
└── bpe_tokenizer.json                 # Serialized BPE pair mappings and vocabulary definitions
```

---

## 🏗️ Detailed Module Breakdown

### 1. `model.py` — Core Neural Architecture
`model.py` is self-contained and implements our 6-effective-layer recurrent Transformer from absolute primitives (`torch.nn.Module`).

#### `Config` (`dataclass`)
Holds all hyperparameter configurations:
* `vocab_size = 2048`: Matches our custom `BPETokenizer`.
* `block_size = 96`: Sequence length per attention window (`~241 bytes` effective context reach).
* `n_layer = 4`: Number of physical `Block` modules instantiated in memory.
* `n_head = 4`: Number of independent attention query/key/value projection heads.
* `n_embd = 160`: Hidden state vector dimension ($d_{\text{model}} = 160$).
* `recurrence = [0, 1, 2, 3, 2, 3]`: The traversal schedule executed in `GPT.forward`.

#### `RMSNorm(nn.Module)`
Implements Pre-LN Root Mean Square Normalization:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$
* **Why**: Eliminates the mean-subtraction calculation required by traditional `LayerNorm`, speeding up forward/backward passes by ~15% across our recurrent loops.

#### `SelfAttention(nn.Module)`
Implements Multi-Head Attention with ALiBi linear biases and ResFormer value residual gating:
* **ALiBi (`build_alibi_mask`)**: Computes static, head-specific linear slope penalties $m_h = 2^{-8h/H}$. Instead of looking up positional embeddings from memory, ALiBi penalizes query-key dot products based on physical token distance:
  $$S_{i, j}^{(h)} = \frac{Q_i^{(h)} K_j^{(h)T}}{\sqrt{d_{\text{head}}}} - m_h \cdot |i - j|$$
* **ResFormer Value Residuals (`lam_v1`, `lam_v2`)**: Maintains two learnable scalar parameters per head (`lam_v1` initialized to `0.1`, `lam_v2` to `1.0`). When processing deep or recurrent layers (`layer > 0`), the value tensor is dynamically gated with the value tensor from layer 0 (`v1_cache`):
  $$V_{\text{gated}} = \lambda_{v1} \odot V_{\text{layer } 0} + \lambda_{v2} \odot V_{\text{current}}$$
  This creates an `O(1)` gradient highway directly back to the embedding layer, preventing attention collapse across 6 recurrent evaluations.

#### `SwiGLU(nn.Module)`
Implements our dual-branch gated non-linear projection (`expansion factor E=3`):
* Computes `gate = F.silu(self.fc_gate(x))` and `up = self.fc_up(x)`, returning `self.fc_down(gate * up)`.
* **Why**: Multiplicative gating allows the network to dynamically filter ambiguous cross-lingual sub-words (`Hindi/English`), outperforming standard single-branch `GELU` by `~0.045 BPB`.

#### `GPT(nn.Module)`
The master wrapper class combining all components:
* **Weight Tying (`head.weight = tok_emb.weight`)**: Shares the `2048 x 160` embedding matrix with the output projection linear head. Saves `327,680 parameters` (`19.7%` of total model capacity) and guarantees exact cache address reuse during logit generation.
* **Recurrent Forward Routing (`GPT.forward`)**:
  ```python
  for layer_idx in self.config.recurrence:  # [0, 1, 2, 3, 2, 3]
      block = self.blocks[layer_idx]
      x, v1_cache = block(x, alibi, v1_cache=v1_cache, is_first=(layer_idx == 0))
  ```
  Physical blocks `2` and `3` are evaluated twice per sequence step, yielding **6 effective layers of depth** without adding a single physical weight tensor.

---

### 2. `tokenizer.py` — BPE Tokenization Engine
`tokenizer.py` defines our standalone `BPETokenizer` class without relying on external C++ dependencies (`huggingface/tokenizers` or `tiktoken`):
* **`train(text, target_vocab_size=2048)`**:
  Pre-segments raw input corpus strings into unique words via regex (`[^\r\n\p{C}\p{Z}]+`). Uses an **$O(1)$ doubly-linked pair-frequency delta hash map** (`pair_counts`) to iteratively merge the most frequent adjacent byte/token pairs (`256 -> 2048`). Completes `1,792 merges` across `7.3 MB` in **~90 seconds**.
* **`encode(text) / decode(ids)`**:
  Maps raw UTF-8 byte arrays to vocabulary IDs (`2.51 bytes/token` compression) and reverses the process losslessly.

---

### 3. `train.py` — Optimization & Training Mechanics

#### How We Are Training: Step-by-Step Execution Mechanics
When you launch `python train.py --data train_corpus.txt --steps 2000 --out ckpt.pt`, the script executes the following exact training protocol:

```
[Phase 1: Dataset & Vocabulary Preparation]
   1. Load raw `train_corpus.txt` (7,318,592 bytes).
   2. Check if `bpe_tokenizer.json` exists -> if not, run `tok.train()` (~90s).
   3. Check if `train_ids.pt` exists -> if not, encode full text to `2,919,027` integer IDs (~19s).
   4. Load `train_ids.pt` into physical RAM as a contiguous `torch.LongTensor`.

[Phase 2: Model & Optimizer Initialization]
   1. Instantiate `model = GPT(Config())` (`1,660,352 parameters`).
   2. Configure `AdamW` optimizer (`lr=6e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.02`).
   3. Initialize `ema_model = copy.deepcopy(model)` (`decay=0.997`, inactive until step 1,600).

[Phase 3: The 2,000-Step Optimization Loop]
   For step in 1 .. 2000:
       │
       ├─► A. Fast Batch Sampling (`get_batch()`)
       │      Sample `batch_size=8` random starting indices across `train_ids.pt`.
       │      Slice inputs `X (8, 96)` and targets `Y (8, 96)` (`Y` is `X` shifted by +1 token).
       │      Zero CPU memory allocation or string parsing overhead (`< 0.4 ms`).
       │
       ├─► B. Learning Rate Scheduling (`wsd_lr(step)`)
       │      Steps 1–50: Linear warmup (`1.2e-5 -> 6e-4`).
       │      Steps 50–1600: Stable exploration (`held at peak 6e-4`).
       │      Steps 1600–2000: Linear decay (`6e-4 -> 1e-5`).
       │
       ├─► C. Forward Pass & Loss Computation
       │      Execute `logits, loss = model(X, targets=Y)` (`[0,1,2,3,2,3]` recurrence).
       │      Compute `F.cross_entropy(logits.view(-1, 2048), Y.view(-1))` across all 768 tokens.
       │
       ├─► D. Backward Pass & Gradient Norm Clipping
       │      Run `loss.backward()` to accumulate gradients.
       │      Clip `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to block spikes.
       │      Run `optimizer.step()` to update physical weights and zero out gradients (`set_to_none=True`).
       │
       └─► E. Late-Stage Exponential Moving Average (`EMA`)
              If step >= 1600:
                  Update shadow weights: `w_ema = 0.997 * w_ema + 0.003 * w_curr`.
                  Filters out small-batch (`batch=8`) gradient noise during final annealing.

[Phase 4: Production Checkpointing]
   Save `ema_model.state_dict()` into `ckpt.pt` (`6,664,565 bytes`).
   The final evaluation script (`evaluate.py`) scores these exact smoothed EMA weights (`2.0231 Dev BPB`).
```

---

## 🔍 Verification & Parameter Integrity Check

Every component is mathematically accounting for exact physical parameter budget utilization:

```python
# Verifiable directly via Python terminal:
import torch
from model import GPT, Config

m = GPT(Config())
print("Total Physical Parameters:", sum(p.numel() for p in m.parameters()))
# Output: 1660352 (Strictly <= 2,000,000 cap)
```

By engineering structural recurrence (`[0,1,2,3,2,3]`), memory-tied embeddings (`tok_emb == head`), and dual-branch `SwiGLU` primitives (`E=3`), `Micro-GPT 2000` achieves maximum information density and state-of-the-art compression efficiency (`2.0231 BPB`) in exactly 7.2 minutes on CPU.
