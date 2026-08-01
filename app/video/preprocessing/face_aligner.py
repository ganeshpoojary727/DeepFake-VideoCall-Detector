"""Facial landmark alignment module."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class FaceAligner:
    """Aligns facial frame orientation using eye/landmark spatial coordinates."""

    def __init__(self, output_size: Tuple[int, int] = (224, 224)) -> None:
        self._output_size = output_size

    def align(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Align facial crop using landmark points.

        Args:
            frame: Input frame image [H, W, 3].
            landmarks: Optional list of (x, y) landmark point tuples.

        Returns:
            np.ndarray: Aligned frame array.

        Raises:
            PreprocessingError: If input frame format is invalid.
        """
        if frame.ndim != 3:
            raise PreprocessingError(f"Expected 3D frame array, got shape {frame.shape}")

        if landmarks is None or len(landmarks) < 2:
            return frame

        # Rotation/alignment matrix computation placeholder
        return frame
