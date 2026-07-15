"""Fast micro-GPT — optimised for ≤10 min CPU training @ 2000 steps.

Architecture (~1,863,680 params):
  • BPE vocab 2048, d_model=160, 6 layers (straight stack, no recurrence)
  • Weight-tied embeddings, ALiBi (0 positional params), Pre-RMSNorm
  • GeLU MLP with 3× expansion (cheaper per step than SwiGLU)

Dropped vs prior antigravity (speed > marginal BPB):
  depth recurrence, value residuals, SwiGLU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size   = 2048
    block_size   = 96
    n_layer      = 4
    n_head       = 4
    n_embd       = 160
    mlp_ratio    = 3
    dropout      = 0.0
    tie_weights  = True
    recurrence   = [0, 1, 2, 3, 2, 3]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])


def build_alibi_mask(block_size: int, n_heads: int) -> torch.Tensor:
    slopes = _alibi_slopes(n_heads)
    pos = torch.arange(block_size)
    rel = (pos.unsqueeze(1) - pos.unsqueeze(0)).float()
    bias = -slopes.view(-1, 1, 1) * rel.abs().unsqueeze(0)
    causal = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
    bias.masked_fill_(causal.unsqueeze(0), float("-inf"))
    return bias.unsqueeze(0)


class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, alibi: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=alibi[:, :, :T, :T])
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        inner = cfg.mlp_ratio * cfg.n_embd
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.fc1 = nn.Linear(cfg.n_embd, inner, bias=False)
        self.fc2 = nn.Linear(inner, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, alibi: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), alibi)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.register_buffer(
            "alibi_mask",
            build_alibi_mask(cfg.block_size, cfg.n_head),
            persistent=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.alibi_mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
