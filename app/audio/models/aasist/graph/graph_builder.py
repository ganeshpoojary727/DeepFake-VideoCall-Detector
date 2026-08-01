"""Graph builder module for converting latent feature maps into graph representations.

Provides GraphBuilder for orchestrating node feature generation and dynamic adjacency
matrix construction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from app.audio.models.aasist.graph.adjacency import AdjacencyMatrixBuilder
from app.audio.models.aasist.graph.node_features import NodeFeatureGenerator


class GraphBuilder(nn.Module):
    """Orchestrates node feature extraction and dynamic graph adjacency construction.

    Args:
        in_dim (int): Dimensionality of input feature representations.
        node_dim (int): Target node feature vector dimension.
        metric (str): Adjacency similarity metric ('cosine', 'dot', 'euclidean').
    """

    def __init__(
        self,
        in_dim: int = 128,
        node_dim: int = 64,
        metric: str = "cosine",
    ) -> None:
        super().__init__()
        self.node_generator = NodeFeatureGenerator(
            in_dim=in_dim,
            node_dim=node_dim,
            normalize=True,
        )
        self.adjacency_builder = AdjacencyMatrixBuilder(metric=metric)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert input latent features into graph nodes H and adjacency matrix A.

        Args:
            features (torch.Tensor): Feature tensor of shape (batch, channels, length) or (batch, length, channels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (nodes H, adjacency A) tuple.
                - nodes H: shape (batch, num_nodes, node_dim)
                - adjacency A: shape (batch, num_nodes, num_nodes)
        """
        nodes = self.node_generator(features)
        adj = self.adjacency_builder(nodes)
        return nodes, adj
