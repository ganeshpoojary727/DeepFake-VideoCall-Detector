"""Video array to PyTorch tensor converter module."""

from __future__ import annotations

from typing import List
import numpy as np
import torch

from app.video.exceptions.video_exceptions import PreprocessingError


class VideoTensorConverter:
    """Converts numpy frame arrays to normalized float PyTorch video tensors [T, C, H, W]."""

    def __init__(self, scale_to_unit: bool = True) -> None:
        self._scale_to_unit = scale_to_unit

    def to_tensor(self, frames: np.ndarray | List[np.ndarray]) -> torch.Tensor:
        """Convert sequence frames to PyTorch video tensor [T, C, H, W].

        Args:
            frames: Numpy array [T, H, W, C] or list of [H, W, C] arrays.

        Returns:
            torch.Tensor: PyTorch float tensor with shape [T, C, H, W].

        Raises:
            PreprocessingError: If input sequence format is invalid.
        """
        if isinstance(frames, list):
            if not frames:
                raise PreprocessingError("Cannot convert empty frame list to tensor.")
            arr = np.stack(frames, axis=0)  # [T, H, W, C]
        elif isinstance(frames, np.ndarray):
            arr = frames
            if arr.ndim == 3:
                arr = np.expand_dims(arr, axis=0)  # [1, H, W, C]
        else:
            raise PreprocessingError(f"Unsupported input type {type(frames)}")

        if arr.ndim != 4:
            raise PreprocessingError(f"Expected 4D array [T, H, W, C], got shape {arr.shape}")

        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # [T, C, H, W]

        if self._scale_to_unit and tensor.max() > 1.0:
            tensor = tensor / 255.0

        return tensor
