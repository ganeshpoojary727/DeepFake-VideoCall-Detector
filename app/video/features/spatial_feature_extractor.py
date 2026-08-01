"""Spatial feature extractor abstract interface module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch

from app.video.features.base_feature_extractor import BaseFeatureExtractor


class SpatialFeatureExtractor(BaseFeatureExtractor, ABC):
    """Interface for frame-level 2D spatial CNN feature extraction."""

    def __init__(self, out_dim: int = 1792) -> None:
        super().__init__(out_dim=out_dim)

    @abstractmethod
    def extract_spatial_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract spatial embeddings for each frame [B*T, C, H, W] -> [B, T, D].

        Args:
            frames: Video frames tensor [B, T, C, H, W].

        Returns:
            torch.Tensor: Spatial feature embeddings tensor [B, T, D].
        """
        pass

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of BaseFeatureExtractor contract."""
        return self.extract_spatial_features(x)
