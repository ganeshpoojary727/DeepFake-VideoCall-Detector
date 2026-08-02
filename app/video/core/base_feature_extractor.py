"""Base feature extractor interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class BaseFeatureExtractor(ABC):
    """Abstract base class for spatial and temporal feature extraction modules."""

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from preprocessed video frame tensors.

        Args:
            x: Input video tensor [B, T, C, H, W] or [B, C, H, W].

        Returns:
            torch.Tensor: Extracted feature embedding tensor [B, D] or [B, T, D].
        """
        pass


# Base class alias
BaseVideoFeatureExtractor = BaseFeatureExtractor
