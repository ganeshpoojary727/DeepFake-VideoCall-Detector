"""Temporal sequence pooling module supporting Mean, Max, CLS Token, and Attention Pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPooling(nn.Module):
    """Sequence aggregation module mapping (B, T, D) to a single clip embedding (B, D)."""

    def __init__(self, feature_dim: int = 1792, pooling_type: str = "attention") -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pooling_type = pooling_type.lower().strip()

        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        elif self.pooling_type == "attention":
            self.attn_query = nn.Linear(feature_dim, 1, bias=False)

    def prepend_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend learnable CLS token to sequence if pooling_type is 'cls'."""
        if self.pooling_type == "cls":
            b = x.size(0)
            cls_tokens = self.cls_token.expand(b, -1, -1)
            return torch.cat((cls_tokens, x), dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence tensor of shape (B, T, D) or (B, T+1, D) if CLS token was prepended.

        Returns:
            torch.Tensor: Single aggregated clip embedding of shape (B, D).
        """
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        elif self.pooling_type == "max":
            return x.max(dim=1)[0]
        elif self.pooling_type == "cls":
            return x[:, 0, :]
        elif self.pooling_type == "attention":
            # Attention pooling weights: Softmax(W @ x)
            attn_weights = F.softmax(self.attn_query(x), dim=1)  # (B, T, 1)
            return torch.sum(attn_weights * x, dim=1)  # (B, D)
        else:
            return x.mean(dim=1)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal attention weights over sequence frames.

        Args:
            x: Input sequence tensor of shape (B, T, D).

        Returns:
            torch.Tensor: Attention weights of shape (B, T, 1) summing to 1.0 over T.
        """
        if self.pooling_type == "attention":
            return F.softmax(self.attn_query(x), dim=1)
        else:
            b, t = x.size(0), x.size(1)
            weights = torch.full((b, t, 1), 1.0 / t, device=x.device, dtype=x.dtype)
            return weights

