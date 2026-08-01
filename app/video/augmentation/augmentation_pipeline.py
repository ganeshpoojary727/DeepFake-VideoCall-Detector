"""Composite video augmentation pipeline container module."""

from __future__ import annotations

from typing import List
import torch

from app.video.augmentation.base_augmentation import BaseVideoAugmentation


class VideoAugmentationPipeline:
    """Composes multiple spatial and temporal video augmentations into a sequence chain."""

    def __init__(self, transforms: List[BaseVideoAugmentation]) -> None:
        self._transforms = transforms

    @property
    def transforms(self) -> List[BaseVideoAugmentation]:
        """Get registered transform list."""
        return self._transforms

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply all transforms sequentially onto video tensor [T, C, H, W]."""
        for t in self._transforms:
            video = t(video)
        return video
