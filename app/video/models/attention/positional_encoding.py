"""Learnable 1D temporal positional encoding for frame sequence embeddings."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal 1D temporal positional encoding supporting variable sequence lengths."""

    def __init__(self, max_len: int = 128, feature_dim: int = 1792, dropout: float = 0.1) -> None:
        super().__init__()
        self.max_len = max_len
        self.feature_dim = feature_dim

        pe = torch.zeros(max_len, feature_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2, dtype=torch.float) * (-math.log(10000.0) / feature_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, feature_dim)

        self.register_buffer("pos_embedding", pe)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to sequence embeddings.

        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Tensor with positional encoding added, shape (B, T, D).
        """
        b, t, d = x.shape
        if t <= self.max_len:
            pos = self.pos_embedding[:, :t, :]
        else:
            # Interpolate positional embeddings for sequence lengths exceeding max_len
            pos = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=t,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        return self.drop(x + pos)
