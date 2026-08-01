"""AASIST back-end graph attention and classification module.

Provides GraphAttentionLayer and AASISTBackEnd for executing graph attention,
readout pooling, and classification projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer implementing batched graph message passing.

    Args:
        in_features (int): Input node feature dimension.
        out_features (int): Output node feature dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Execute graph message passing H' = LeakyReLU(A * H * W).

        Args:
            nodes (torch.Tensor): Node embeddings of shape (batch, num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (batch, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Updated node embeddings of shape (batch, num_nodes, out_features).
        """
        # Linear projection: H * W
        h_proj = self.linear(nodes)

        # Graph message passing: A * (H * W)
        out = torch.bmm(adj, h_proj)
        out = self.act(out)
        out = self.dropout(out)
        return out


class AASISTBackEnd(nn.Module):
    """Back-end module for Graph Attention, Readout Pooling, and Classification projection.

    Args:
        node_dim (int): Input graph node feature dimension.
        num_classes (int): Number of output classification targets.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        node_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gat1 = GraphAttentionLayer(node_dim, node_dim, dropout=dropout)
        self.gat2 = GraphAttentionLayer(node_dim, node_dim, dropout=dropout)

        # Readout pooling: Max and Mean pooling concatenation
        self.fc_pool = nn.Linear(node_dim * 2, node_dim)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(node_dim, num_classes)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Process graph nodes and adjacency matrix into logits.

        Args:
            nodes (torch.Tensor): Node embeddings of shape (batch, num_nodes, node_dim).
            adj (torch.Tensor): Adjacency matrix of shape (batch, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Classification logits of shape (batch, num_classes).
        """
        # Stacked Graph Attention passes
        h = self.gat1(nodes, adj)
        h = self.gat2(h, adj)

        # Global Readout Pooling (Max + Mean)
        mean_pool = h.mean(dim=1)
        max_pool, _ = h.max(dim=1)
        graph_repr = torch.cat([mean_pool, max_pool], dim=-1)

        # Classifier projection
        feat = self.fc_pool(graph_repr)
        feat = self.act(feat)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits
