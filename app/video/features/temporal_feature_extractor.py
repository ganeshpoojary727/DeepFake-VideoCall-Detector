"""Temporal feature extractor abstract interface module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch

from app.video.features.base_feature_extractor import BaseFeatureExtractor


class TemporalFeatureExtractor(BaseFeatureExtractor, ABC):
    """Interface for cross-frame temporal feature extraction across video sequence."""

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__(out_dim=out_dim)

    @abstractmethod
    def extract_temporal_features(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """Process spatial frame features [B, T, D_in] into temporal representation [B, D_out].

        Args:
            spatial_features: Frame feature embeddings tensor [B, T, D_in].

        Returns:
            torch.Tensor: Aggregated sequence feature tensor [B, D_out].
        """
        pass

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of BaseFeatureExtractor contract."""
        return self.extract_temporal_features(x)
