"""Temporal Feature Extractor wrapper module."""

from __future__ import annotations

from typing import Union
import torch
import torch.nn as nn

from app.video.models.attention.base_attention import BaseTemporalAttention


class TemporalFeatureExtractor(nn.Module):
    """Wrapper exposing standard interfaces for extracting temporal clip embeddings from frame sequence features."""

    def __init__(self, encoder: Union[BaseTemporalAttention, nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder

    def encode(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Encode sequence of frame features into a single temporal clip embedding.

        Args:
            sequence_features: Input frame embeddings tensor of shape (B, T, 1792).

        Returns:
            torch.Tensor: Single clip embedding of shape (B, 1792).
        """
        if hasattr(self.encoder, "aggregate"):
            return self.encoder.aggregate(sequence_features)
        return self.encoder(sequence_features)

    def extract_clip_embedding(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Alias for encode method."""
        return self.encode(sequence_features)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to encode."""
        return self.encode(sequence_features)
