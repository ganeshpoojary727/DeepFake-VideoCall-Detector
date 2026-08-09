"""Video tensor normalization module."""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch

from app.video.constants.video_constants import IMAGENET_MEAN, IMAGENET_STD
from app.video.exceptions.video_exceptions import PreprocessingError


class VideoNormalizer:
    """Normalizes video frame tensors using specified mean and standard deviation."""

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self._mean = torch.tensor(mean or IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(std or IMAGENET_STD).view(1, 3, 1, 1)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize video tensor [T, C, H, W] or [B, T, C, H, W].

        Args:
            tensor: PyTorch tensor with channel dimension = 3.

        Returns:
            torch.Tensor: Normalized PyTorch tensor.

        Raises:
            PreprocessingError: If tensor rank is invalid.
        """
        if tensor.dim() not in (4, 5):
            raise PreprocessingError(
                f"Expected 4D [T, C, H, W] or 5D [B, T, C, H, W] tensor, got {tensor.shape}"
            )

        # Move mean and std to tensor device
        mean = self._mean.to(tensor.device, tensor.dtype)
        std = self._std.to(tensor.device, tensor.dtype)

        if tensor.dim() == 4:  # [T, C, H, W]
            # Permute mean for 4D broadcast: [1, 3, 1, 1]
            norm = (tensor - mean) / std
        else:  # [B, T, C, H, W]
            mean_5d = mean.unsqueeze(0)  # [1, 1, 3, 1, 1]
            std_5d = std.unsqueeze(0)
            norm = (tensor - mean_5d) / std_5d

        return norm
