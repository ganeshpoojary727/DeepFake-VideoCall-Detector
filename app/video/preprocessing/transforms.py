"""Video PyTorch tensor transformation pipeline module."""

from __future__ import annotations

from typing import List, Optional
import torch

from app.video.constants.video_constants import IMAGENET_MEAN, IMAGENET_STD
from app.video.preprocessing.video_normalizer import VideoNormalizer


class VideoTransforms:
    """Standard video PyTorch tensor preprocessing transformation pipeline."""

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        normalize: bool = True,
    ) -> None:
        self.normalizer = VideoNormalizer(mean=mean or list(IMAGENET_MEAN), std=std or list(IMAGENET_STD))
        self.normalize = normalize

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Process video tensor [T, C, H, W].

        Args:
            video: Input PyTorch tensor.

        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        if video.dtype != torch.float32:
            video = video.float()
        if video.max() > 1.0:
            video = video / 255.0

        if self.normalize:
            video = self.normalizer.normalize(video)

        return video
