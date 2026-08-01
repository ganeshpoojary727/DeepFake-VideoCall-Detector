"""Composite audio augmentation pipeline container module."""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn

from app.audio.augmentation.audio_augmentations import BaseAudioAugmentation


class AudioAugmentationPipeline(nn.Module):
    """Composes multiple independent audio augmentations into a transform chain."""

    def __init__(self, transforms: List[BaseAudioAugmentation]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all transforms sequentially onto audio tensor."""
        for t in self.transforms:
            x = t(x)
        return x
