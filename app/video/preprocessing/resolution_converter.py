"""Spatial frame resolution conversion module."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class ResolutionConverter:
    """Resizes video frames spatially to target (height, width) dimensions."""

    def __init__(self, target_resolution: Tuple[int, int] = (224, 224)) -> None:
        self._target_resolution = target_resolution

    def convert(self, frame: np.ndarray) -> np.ndarray:
        """Resize single frame array [H, W, C] to target resolution.

        Args:
            frame: Input 3D frame numpy array.

        Returns:
            np.ndarray: Resized 3D frame numpy array.

        Raises:
            PreprocessingError: If input frame format is invalid.
        """
        if frame.ndim != 3:
            raise PreprocessingError(f"Expected 3D frame array, got shape {frame.shape}")

        target_h, target_w = self._target_resolution
        orig_h, orig_w = frame.shape[:2]

        if orig_h == target_h and orig_w == target_w:
            return frame

        # Nearest-neighbor spatial interpolation fallback without requiring OpenCV
        row_indices = np.linspace(0, orig_h - 1, target_h, dtype=int)
        col_indices = np.linspace(0, orig_w - 1, target_w, dtype=int)

        resized = frame[row_indices[:, None], col_indices]
        return resized

    def convert_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Resize a list of frame arrays spatially."""
        return [self.convert(f) for f in frames]
