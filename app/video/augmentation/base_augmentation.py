"""Base video augmentation class abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch

from app.video.exceptions.video_exceptions import AugmentationError


class BaseVideoAugmentation(ABC):
    """Abstract base class for spatial and temporal video tensor augmentations.

    Operates on PyTorch video tensors with shape [T, C, H, W] or [B, T, C, H, W].
    """

    def __init__(self, p: float = 0.5) -> None:
        if not (0.0 <= p <= 1.0):
            raise AugmentationError(f"Probability p must be in [0, 1], got {p}")
        self._p = p

    @property
    def probability(self) -> float:
        """Get application probability."""
        return self._p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply augmentation conditionally based on probability p."""
        if torch.rand(1).item() < self._p:
            return self.apply(video)
        return video

    @abstractmethod
    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply video transformation logic.

        Args:
            video: PyTorch tensor [T, C, H, W].

        Returns:
            torch.Tensor: Augmented video tensor [T, C, H, W].
        """
        pass
