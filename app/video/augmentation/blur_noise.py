"""Spatial blur and noise video augmentations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from app.video.augmentation.base_augmentation import BaseVideoAugmentation


class Blur(BaseVideoAugmentation):
    """Applies spatial smoothing blur to video frames."""

    def __init__(self, kernel_size: int = 5, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._kernel_size = kernel_size

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply spatial mean filter across video frame sequence."""
        k = self._kernel_size
        padding = k // 2
        return F.avg_pool2d(video, kernel_size=k, stride=1, padding=padding)


class GaussianBlur(Blur):
    """Applies Gaussian spatial blur to video frames."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, p: float = 0.5) -> None:
        super().__init__(kernel_size=kernel_size, p=p)
        self._sigma = sigma


class GaussianNoise(BaseVideoAugmentation):
    """Applies additive Gaussian noise to frame pixels."""

    def __init__(self, std: float = 0.05, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._std = std

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Add normal distribution noise tensor."""
        noise = torch.randn_like(video) * self._std
        return torch.clamp(video + noise, 0.0, 1.0)


class Noise(GaussianNoise):
    """Generic noise augmentation alias for GaussianNoise."""

    pass
