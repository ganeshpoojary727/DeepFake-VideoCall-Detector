"""Base video feature extractor interface abstraction module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseFeatureExtractor(nn.Module, ABC):
    """Abstract base class for all video feature extraction modules."""

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self._out_dim = out_dim

    @property
    def output_dim(self) -> int:
        """Get output feature representation dimensionality."""
        return self._out_dim

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representation tensors from input video frames.

        Args:
            x: Input video tensor [B, T, C, H, W] or [T, C, H, W].

        Returns:
            torch.Tensor: Feature map tensor.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch nn.Module forward pass redirecting to extract_features."""
        return self.extract_features(x)
