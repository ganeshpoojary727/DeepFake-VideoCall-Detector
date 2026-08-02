"""Multi-head self-attention module for temporal frame sequence modeling."""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMultiHeadAttention(nn.Module):
    """Multi-head self-attention operating on sequence embeddings of shape (B, T, D)."""

    def __init__(
        self,
        feature_dim: int = 1792,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(feature_dim, feature_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence tensor of shape (B, T, D).
            mask: Optional boolean or attention mask.

        Returns:
            torch.Tensor: Attended sequence tensor of shape (B, T, D).
        """
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, T, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v).transpose(1, 2).reshape(b, t, d)
        out = self.proj(x_attn)
        return self.proj_drop(out)
