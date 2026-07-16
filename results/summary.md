# Run Summary — Restored Full Architecture

## Training (`train.py`)

| Setting | Value |
|---|---|
| Steps | 2000 |
| Batch size | 8 |
| Block size | 96 |
| Peak LR | 6e-4 |
| Warmup steps | 50 |
| LR schedule | WSD (Warmup-Stable-Decay) + EMA (decay=0.997 from step 1600) |
| Weight decay | 0.02 (epoch_ratio=0.53) |
| Architecture | 4 physical blocks → 6 effective layers (`[0, 1, 2, 3, 2, 3]`), SwiGLU ($E=3$), ResFormer |
| Corpus | 7,318,592 bytes → 2,919,027 tokens (2.51 bytes/token, vocab=2048) |
| Parameters | 1,660,352 |
| Wall time | 435s = **7.2 min** |
| Checkpoint | `ckpt.pt` |

### Loss Curve

| Step | Avg Loss | LR | Speed |
|---|---|---|---|
| 1 | 7.7616 | 0.000012 | 234 ms/step |
| 100 | 5.7613 | 0.000600 | 244 ms/step |
| 200 | 4.6826 | 0.000600 | 235 ms/step |
| 500 | 4.1537 | 0.000600 | 229 ms/step |
| 1000 | 3.8316 | 0.000600 | 219 ms/step |
| 1500 | 3.5699 | 0.000600 | 215 ms/step |
| 1800 | 3.4339 | 0.000305 | 219 ms/step |
| 2000 | 3.3289 | 0.000010 | 217 ms/step |

---

## Evaluation (`evaluate.py`)

| Metric | Value |
|---|---|
| **BPB (Bits Per Byte)** | **2.0231** |
| Parameters | 1,660,352 |
| Steps | 2000 |
| Tokens in eval file | 61,401 |
| Tokens scored | 61,400 |

---

## Files in Submission Folder

| File | Description |
|---|---|
| `ckpt.pt` | Final trained weights (`1,660,352 params`, `2.0231 BPB`) |
| `model.py` | Full architecture definition (SwiGLU, `[0,1,2,3,2,3]` recurrence, ResFormer, ALiBi, RMSNorm) |
| `train.py` | Full training script (WSD schedule, AdamW, EMA, gradient clipping, token cache) |
| `tokenizer.py` | Self-contained BPE tokenizer (`V=2048`) |
| `evaluate.py` | Unchanged evaluator required by grading |
| `RUNLOG.md` | Graded run log detailing Runs 0 through 4 |
| `NOTES.md` | Technical rationale (max 10 sentences) |
| `SUMMARY.html` | Rich interactive HTML dashboard summarizing the complete project |
