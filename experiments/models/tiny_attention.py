"""Minimal 2-layer causal self-attention model -- the literature's known-sufficient
reference architecture for associative recall / induction-head-style tasks (Olsson et al.
2022 "In-context Learning and Induction Heads"; Elhage et al. 2021 "A Mathematical
Framework for Transformer Circuits", where induction heads were first identified in
exactly this class of toy 2-layer attention model). Built for thread 18
(docs/threads/18-recall-protocol-validation.md): does the exact recall protocol used by
threads 9-17 admit a solution at all, by a reference architecture the literature says
should solve it easily at this scale?

Standard PyTorch default init throughout (no muP scaling, no repo-specific tuning) --
deliberately the most textbook-standard construction available, so a failure can't be
blamed on an idiosyncratic init choice the way thread 17's untrained-conv confound was.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (batch, n_heads, seq_len, head_dim)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (batch, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)), then x + MLP(LN(x)) -- standard GPT-style ordering."""

    def __init__(self, hidden: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, n_heads)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyAttentionModel(nn.Module):
    """Embedding + learned positional embedding -> n_layers causal transformer blocks ->
    final LayerNorm -> readout at the final (query) position. `seq_len` fixes the
    positional-embedding table size, so it must match the task's actual sequence length
    (recall.seq_len_for(n_pairs)) -- attention has no recurrence to encode order, so
    without this the model cannot distinguish "the token right after a matched key" from
    any other position (see thread 18's pre-registration, "positional information")."""

    def __init__(self, vocab_size: int, hidden: int, seq_len: int, n_layers: int = 2,
                 n_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden, n_heads, mlp_ratio) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(hidden)
        self.readout = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.readout(x[:, -1, :])
