"""Facial region cropping module."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class FaceCropper:
    """Crops face region bounding boxes from image frames with padding margin."""

    def __init__(self, margin: float = 0.2, target_size: Optional[Tuple[int, int]] = None) -> None:
        self._margin = margin
        self._target_size = target_size

    def crop(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Crop face from frame given bounding box tuple (x1, y1, x2, y2).

        Args:
            frame: Raw image array [H, W, 3].
            bbox: Bounding box coordinates (x1, y1, x2, y2). If None, full frame is returned.

        Returns:
            np.ndarray: Cropped face frame array.

        Raises:
            PreprocessingError: If input frame shape is invalid.
        """
        if frame.ndim != 3:
            raise PreprocessingError(f"Expected 3D frame array [H, W, C], got shape {frame.shape}")

        if bbox is None:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(bw * self._margin)
        pad_y = int(bh * self._margin)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        cropped = frame[cy1:cy2, cx1:cx2]
        if cropped.size == 0:
            return frame

        return cropped
