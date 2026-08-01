"""Temporal frame dropping and skipping video augmentations."""

from __future__ import annotations

import torch

from app.video.augmentation.base_augmentation import BaseVideoAugmentation


class TemporalDrop(BaseVideoAugmentation):
    """Zeros out random frames in temporal sequence to simulate frame dropouts."""

    def __init__(self, drop_prob: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._drop_prob = drop_prob

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Apply zero mask to random temporal frame indices."""
        t = video.shape[0]
        mask = (torch.rand(t) >= self._drop_prob).float().view(t, 1, 1, 1)
        mask = mask.to(video.device)
        return video * mask


class FrameSkip(BaseVideoAugmentation):
    """Sub-samples video frames along temporal dimension by stride step."""

    def __init__(self, stride: int = 2, p: float = 0.5) -> None:
        super().__init__(p=p)
        self._stride = stride

    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """Subsample frames by stride factor and pad back to original length T."""
        t = video.shape[0]
        subsampled = video[:: self._stride]
        sub_t = subsampled.shape[0]
        if sub_t < t:
            pad = subsampled[-1:].repeat(t - sub_t, 1, 1, 1)
            subsampled = torch.cat([subsampled, pad], dim=0)
        return subsampled
