"""Feature fusion module for combining spectro-temporal and graph embeddings.

Provides FeatureFusion for fusing front-end encoder features with graph representations
using residual connections, LayerNorm, and Dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Fuses spectro-temporal front-end features and graph back-end representations.

    Args:
        feature_dim (int): Dimensionality of input feature vectors.
        dropout (float): Dropout probability.
    """

    def __init__(self, feature_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim * 2, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        frontend_features: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse front-end and graph representations with residual shortcut connection.

        Args:
            frontend_features (torch.Tensor): Front-end embedding tensor of shape (batch, feature_dim).
            graph_features (torch.Tensor): Graph embedding tensor of shape (batch, feature_dim).

        Returns:
            torch.Tensor: Fused representation tensor of shape (batch, feature_dim).
        """
        # Ensure dimensions match
        if frontend_features.ndim != graph_features.ndim:
            graph_features = graph_features.view_as(frontend_features)

        # Concatenate features
        cat_feat = torch.cat([frontend_features, graph_features], dim=-1)
        fused = self.proj(cat_feat)
        fused = self.act(fused)
        fused = self.dropout(fused)

        # Residual shortcut connection and LayerNorm
        output = self.norm(frontend_features + fused)
        return output
