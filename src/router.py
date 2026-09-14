"""Dispatch model for ESP32 model dispatch — decides which expert(s) to activate.

Architecture: tiny transformer classifier.
Input: tokenized query (same vocab as experts).
Output: probability distribution over N experts.

The router is small enough to run on a Pi 5 in milliseconds, or on an
ESP32-S3 in SRAM. It sees the first ~64 tokens of the query and routes
to top-k=2 experts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RouterConfig:
    vocab_size: int = 32768
    d_model: int = 64       # tiny — this is a classifier, not a generator
    n_layers: int = 2
    n_heads: int = 2
    n_experts: int = 3
    seq_len: int = 64       # only need enough context to classify domain
    pool: str = "mean"      # how to aggregate token reps → single vector

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class RouterModel(nn.Module):
    """Tiny transformer that classifies which expert should handle a query."""

    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([RouterBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, cfg.n_experts, bias=True)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cfg = self.cfg
        B, T = idx.shape
        assert T <= cfg.seq_len, f"Router expects <= {cfg.seq_len} tokens, got {T}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]

        for block in self.blocks:
            x = block(x)

        x = self.out_norm(x)

        # Pool token reps → single vector
        if cfg.pool == "mean":
            pooled = x.mean(dim=1)
        elif cfg.pool == "cls":
            pooled = x[:, 0]  # first token as CLS
        elif cfg.pool == "last":
            pooled = x[:, -1]
        else:
            pooled = x.mean(dim=1)

        logits = self.classifier(pooled)  # (B, n_experts)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def route(self, idx, top_k=2):
        """Return top-k expert indices and their weights (normalized probs)."""
        logits, _ = self(idx)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        # Renormalize the top-k weights so they sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return topk_idx, topk_probs


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RouterBlock(nn.Module):
    """Minimal transformer block: attention + FFN, pre-norm."""

    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = RouterAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RouterAttention(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))


def param_count(cfg: RouterConfig):
    """Estimate router parameter count."""
    emb = cfg.vocab_size * cfg.d_model + cfg.seq_len * cfg.d_model
    per_block = (
        3 * cfg.d_model * cfg.d_model +  # qkv
        cfg.d_model * cfg.d_model +      # proj
        2 * cfg.d_model * cfg.d_model * 2 +  # ffn (up + down)
        2 * cfg.d_model                  # 2 RMSNorm weights
    )
    blocks = cfg.n_layers * per_block
    out_norm = cfg.d_model
    classifier = cfg.d_model * cfg.n_experts + cfg.n_experts
    total = emb + blocks + out_norm + classifier
    return total


if __name__ == "__main__":
    cfg = RouterConfig()
    model = RouterModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"Router params: {n:,} (estimated: {param_count(cfg):,})")
    print(f"Config: {cfg}")

    # Quick test
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(idx, torch.tensor([0, 1]))
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    topk_idx, topk_probs = model.route(idx, top_k=2)
    print(f"Top-2 experts: {topk_idx}")
    print(f"Top-2 weights: {topk_probs}")