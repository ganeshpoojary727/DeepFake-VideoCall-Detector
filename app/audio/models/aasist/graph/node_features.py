"""Node feature generation module for graph representations.

Provides NodeFeatureGenerator for projecting and normalizing encoder outputs
into graph node feature matrices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeFeatureGenerator(nn.Module):
    """Generates and normalizes node feature matrices from encoder latent representations.

    Args:
        in_dim (int): Dimensionality of input encoder features.
        node_dim (int): Target feature dimension for each graph node.
        normalize (bool): Whether to apply L2 normalization to node embeddings.
    """

    def __init__(
        self,
        in_dim: int = 128,
        node_dim: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.node_dim = node_dim
        self.normalize = normalize

        self.proj = (
            nn.Linear(in_dim, node_dim) if in_dim != node_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input feature map or sequence into graph node embeddings.

        Args:
            x (torch.Tensor): Feature tensor of shape (batch, in_dim, length) or (batch, length, in_dim).

        Returns:
            torch.Tensor: Node feature matrix of shape (batch, num_nodes, node_dim).
        """
        if x.ndim == 3:
            # If shape is (batch, channels, length), transpose to (batch, length, channels)
            if x.shape[1] == self.in_dim:
                nodes = x.transpose(1, 2)
            else:
                nodes = x
        elif x.ndim == 2:
            # If shape is (batch, in_dim), unsqueeze to (batch, 1, in_dim)
            nodes = x.unsqueeze(1)
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {x.shape}")

        # Linear projection
        nodes = self.proj(nodes)

        # L2 Normalization across node feature dimension
        if self.normalize:
            nodes = F.normalize(nodes, p=2, dim=-1)

        return nodes
