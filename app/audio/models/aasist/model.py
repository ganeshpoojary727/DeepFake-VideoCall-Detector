"""Complete AASIST production model implementation.

Provides the AASIST class combining front-end RawNet feature extraction,
dynamic graph building, graph attention, readout pooling, and classification.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from app.audio.core.base_model import BaseAudioModel
from app.audio.models.aasist.graph.graph_builder import GraphBuilder
from app.audio.models.aasist.modules.backend import AASISTBackEnd
from app.audio.models.aasist.modules.frontend import AASISTFrontEnd
from app.audio.models.aasist.modules.fusion import FeatureFusion
from app.audio.registry.model_registry import model_registry


class AASIST(BaseAudioModel):
    """Production AASIST audio deepfake detection model.

    Pipeline:
    audio -> FrontEnd -> GraphBuilder -> GraphAttention -> ReadoutPooling -> Fusion -> Classifier -> Logits

    Args:
        num_classes (int): Number of target classification categories (default 2: Bonafide / Spoof).
        in_channels (int): Input audio channels (default 1).
        sinc_channels (int): Number of SincConv bandpass filter channels.
        node_dim (int): Graph node feature dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        sinc_channels: int = 128,
        node_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Front-end RawNet feature encoder
        self.frontend = AASISTFrontEnd(
            in_channels=in_channels,
            sinc_channels=sinc_channels,
            embedding_dim=node_dim * 2,
        )

        # Graph builder (node generation & dynamic adjacency)
        self.graph_builder = GraphBuilder(
            in_dim=node_dim * 2,
            node_dim=node_dim,
            metric="cosine",
        )

        # Back-end Graph Attention & Readout Pooling classifier
        self.backend = AASISTBackEnd(
            node_dim=node_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Feature fusion
        self.fusion = FeatureFusion(feature_dim=node_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute end-to-end AASIST classification forward pass.

        Args:
            x (torch.Tensor): Raw waveform or spectrogram of shape (batch, 1, length) or (batch, length).

        Returns:
            torch.Tensor: Class logits tensor of shape (batch, num_classes).
        """
        # 1. Front-end feature extraction
        features = self.frontend(x)

        # 2. Dynamic graph construction (nodes H, adjacency A)
        nodes, adj = self.graph_builder(features)

        # 3. Back-end Graph Attention & Readout Pooling classification
        logits = self.backend(nodes, adj)
        return logits

    def get_num_parameters(self) -> int:
        """Calculate total number of trainable model parameters.

        Returns:
            int: Total trainable parameters count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Register AASIST model architecture in model_registry
model_registry.register("aasist", AASIST, overwrite=True)
