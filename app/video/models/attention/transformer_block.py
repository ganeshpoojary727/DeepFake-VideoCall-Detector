"""Transformer Block for temporal feature sequence processing."""

from __future__ import annotations

import torch
import torch.nn as nn
from app.video.models.attention.multihead_attention import TemporalMultiHeadAttention


class TemporalTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with multi-head self-attention and MLP feed-forward network."""

    def __init__(
        self,
        feature_dim: int = 1792,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation_fn: str = "silu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = TemporalMultiHeadAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(feature_dim)
        act_key = activation_fn.lower().strip()
        if act_key == "relu":
            act_layer: nn.Module = nn.ReLU(inplace=True)
        elif act_key == "gelu":
            act_layer = nn.GELU()
        else:
            act_layer = nn.SiLU(inplace=True)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Transformed sequence tensor of shape (B, T, D).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
