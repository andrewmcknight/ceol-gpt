"""
Decoder-only transformer (GPT-2 style, pre-norm) for ceol-gpt.

Architecture:
  - Token embedding + learned positional embedding
  - N x TransformerBlock:
      LayerNorm → CausalSelfAttention → residual
      LayerNorm → FeedForward (GELU)  → residual
  - Final LayerNorm → linear projection to vocab

Pre-norm (LayerNorm before sub-layers, not after) is used throughout for
more stable training from a random initialisation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask: upper-triangular, registered as a buffer (not a parameter)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).bool(),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and split heads
        qkv = self.qkv(x)                          # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)             # each (B, T, C)
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale   # (B, H, T, T)

        # Apply causal mask
        causal = self.causal_mask[:T, :T]
        attn = attn.masked_fill(~causal, float("-inf"))

        # Apply padding mask if provided: attn_mask is (B, T), 1=real 0=pad
        if attn_mask is not None:
            # Expand to (B, 1, 1, T) so padding keys are masked out
            pad_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(pad_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows (can happen at padding positions)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out = attn @ v                              # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One decoder layer: pre-norm self-attention + pre-norm feed-forward."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CeolGPT(nn.Module):
    """Decoder-only transformer for ABC notation generation.

    Forward pass returns logits (B, T, vocab_size).
    Loss is computed externally so this module stays composable.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights.
        # Halves the number of parameters in the embedding/head layers and
        # often improves perplexity on small vocabularies.
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual projections by 1/sqrt(2*n_layers) as in GPT-2
        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, T)
        attn_mask: torch.Tensor | None = None,  # (B, T), 1=real 0=pad
    ) -> torch.Tensor:                    # (B, T, vocab_size)
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.emb_dropout(self.tok_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Factory: build model from config dict (as loaded from YAML)
# ---------------------------------------------------------------------------

def build_model(cfg_dict: dict, vocab_size: int) -> CeolGPT:
    """Construct a CeolGPT from a config dict and vocab size."""
    model_cfg = cfg_dict.get("model", cfg_dict)  # allow flat or nested config
    return CeolGPT(ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=cfg_dict.get("max_seq_len", 512),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    ))


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=930, d_model=512, n_heads=8, n_layers=12, d_ff=2048)
    model = CeolGPT(cfg)
    print(f"Parameters: {model.num_parameters():,}")

    # Forward pass
    B, T = 4, 128
    ids = torch.randint(0, 930, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 100:] = 0  # simulate padding in first sequence

    logits = model(ids, mask)
    print(f"Output shape: {logits.shape}")  # (4, 128, 930)

    # Loss
    targets = torch.randint(0, 930, (B, T))
    loss = F.cross_entropy(logits.view(-1, 930), targets.view(-1), ignore_index=0)
    print(f"Dummy loss: {loss.item():.4f}  (expected ~{math.log(930):.2f})")
