"""Dynamic adjacency matrix construction module for graph attention.

Provides AdjacencyMatrixBuilder for generating batched graph edge adjacency
matrices from node feature embeddings.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdjacencyMatrixBuilder(nn.Module):
    """Dynamically constructs batched graph adjacency matrices from node feature embeddings.

    Args:
        metric (str): Similarity metric for edge weights ('cosine', 'dot', 'euclidean').
        top_k (Optional[int]): Optional top-k sparsification of graph edges per node.
        threshold (Optional[float]): Minimum edge similarity threshold.
    """

    def __init__(
        self,
        metric: str = "cosine",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.metric = metric.lower().strip()
        self.top_k = top_k
        self.threshold = threshold

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Construct batched adjacency matrix A from node features H.

        Args:
            nodes (torch.Tensor): Node embeddings matrix of shape (batch, num_nodes, node_dim).

        Returns:
            torch.Tensor: Adjacency matrix A of shape (batch, num_nodes, num_nodes).
        """
        batch_size, num_nodes, _ = nodes.shape

        if self.metric == "cosine":
            norm_nodes = F.normalize(nodes, p=2, dim=-1)
            adj = torch.bmm(norm_nodes, norm_nodes.transpose(1, 2))
        elif self.metric == "dot":
            adj = torch.bmm(nodes, nodes.transpose(1, 2))
        elif self.metric == "euclidean":
            # Distance matrix computation
            dist = torch.cdist(nodes, nodes, p=2)
            adj = torch.exp(-dist)
        else:
            raise ValueError(f"Unsupported adjacency metric: '{self.metric}'")

        # Thresholding
        if self.threshold is not None:
            adj = torch.where(adj >= self.threshold, adj, torch.zeros_like(adj))

        # Top-K sparsification
        if self.top_k is not None and 0 < self.top_k < num_nodes:
            topk_vals, _ = torch.topk(adj, k=self.top_k, dim=-1)
            min_val = topk_vals[:, :, -1:].expand_as(adj)
            adj = torch.where(adj >= min_val, adj, torch.zeros_like(adj))

        # Softmax normalization across rows
        adj = F.softmax(adj, dim=-1)
        return adj
