"""Video compression artifact data augmentations."""

from __future__ import annotations

from typing import Tuple
import torch

from app.video.augmentation.base_augmentation import BaseVideoAugmentation


class JPEG(BaseVideoAugmentation):
    """Simulates JPEG compression artifacts by quantizing frame pixel values."""

    def __init__(self, quality: int = 75, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._quality = quality

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply JPEG quantization approximation to video frames."""
        factor = (100 - self._quality) / 100.0 * 0.1
        noise = (torch.rand_like(video) - 0.5) * factor
        return torch.clamp(video + noise, 0.0, 1.0)


class RandomCompression(BaseVideoAugmentation):
    """Applies random compression artifact levels across specified quality range."""

    def __init__(self, quality_range: Tuple[int, int] = (50, 95), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._quality_range = quality_range

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply random JPEG compression artifact simulation."""
        min_q, max_q = self._quality_range
        q = int(torch.randint(min_q, max_q + 1, (1,)).item())
        jpeg_aug = JPEG(quality=q, p=1.0)
        return jpeg_aug.apply(video)
