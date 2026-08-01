"""Spatial cropping and flipping video frame augmentations."""

from __future__ import annotations

from typing import Tuple
import torch

from app.video.augmentation.base_augmentation import BaseVideoAugmentation


class HorizontalFlip(BaseVideoAugmentation):
    """Flips all frames in video sequence horizontally along width axis."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Flip tensor along spatial width dimension W (dim=-1)."""
        return torch.flip(video, dims=[-1])


class RandomCrop(BaseVideoAugmentation):
    """Crops video frames spatially at random crop location across sequence."""

    def __init__(self, crop_size: Tuple[int, int] = (224, 224), p: float = 0.5) -> None:
        super().__init__(p=p)
        self._crop_size = crop_size

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply spatial random crop."""
        ch, cw = self._crop_size
        h, w = video.shape[-2:]
        if h <= ch or w <= cw:
            return video

        top = torch.randint(0, h - ch + 1, (1,)).item()
        left = torch.randint(0, w - cw + 1, (1,)).item()
        return video[..., top : top + ch, left : left + cw]
