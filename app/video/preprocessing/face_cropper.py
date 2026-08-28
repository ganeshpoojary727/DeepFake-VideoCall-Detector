"""Facial region cropping and alignment module for video deepfake detection."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import cv2
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class FaceCropper:
    """Crops and normalizes facial region bounding boxes with safety margins and aspect preservation."""

    def __init__(
        self,
        margin: float = 0.2,
        target_size: Optional[Tuple[int, int]] = (224, 224),
    ) -> None:
        self._margin = margin
        self._target_size = target_size

    @property
    def target_size(self) -> Optional[Tuple[int, int]]:
        """Output resolution (width, height)."""
        return self._target_size

    def get_expanded_bbox(
        self,
        frame_shape: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """Compute boundary-clamped margin-expanded bounding box (x1, y1, x2, y2)."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = int(bw * self._margin)
        pad_y = int(bh * self._margin)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        return (cx1, cy1, cx2, cy2)

    def crop(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Crop face from frame given bounding box tuple (x1, y1, x2, y2) and resize to target resolution.

        Args:
            frame: Raw image array [H, W, 3] or [H, W].
            bbox: Bounding box coordinates (x1, y1, x2, y2). If None, center-crop is applied.
            target_size: Optional override (width, height), e.g. (224, 224).

        Returns:
            np.ndarray: Cropped and resized face image array.
        """
        cropped, _ = self.crop_with_bbox_metadata(frame, bbox=bbox, target_size=target_size)
        return cropped

    def crop_with_bbox_metadata(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop face and return cropped image alongside the exact bounding box in the source frame.

        Args:
            frame: Raw image array.
            bbox: (x1, y1, x2, y2) in source coordinates.
            target_size: Optional target size override.

        Returns:
            Tuple[np.ndarray, Tuple[int, int, int, int]]:
                (cropped_resized_frame, (x1, y1, x2, y2))
        """
        if frame.ndim not in (2, 3):
            raise PreprocessingError(f"Expected 2D or 3D frame array, got shape {frame.shape}")

        h, w = frame.shape[:2]
        out_size = target_size or self._target_size

        if bbox is not None:
            cx1, cy1, cx2, cy2 = self.get_expanded_bbox((h, w), bbox)
            cropped = frame[cy1:cy2, cx1:cx2]
            actual_box = (cx1, cy1, cx2, cy2)
        else:
            # Center-crop square fallback
            short_side = min(h, w)
            cy, cx = h // 2, w // 2
            cy1 = max(0, cy - short_side // 2)
            cy2 = min(h, cy + short_side // 2)
            cx1 = max(0, cx - short_side // 2)
            cx2 = min(w, cx + short_side // 2)
            cropped = frame[cy1:cy2, cx1:cx2]
            actual_box = (cx1, cy1, cx2, cy2)

        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = frame
            actual_box = (0, 0, w, h)

        if out_size is not None:
            tw, th = out_size
            if cropped.shape[1] != tw or cropped.shape[0] != th:
                interp = cv2.INTER_AREA if (cropped.shape[1] > tw and cropped.shape[0] > th) else cv2.INTER_LINEAR
                cropped = cv2.resize(cropped, (tw, th), interpolation=interp)

        return cropped, actual_box
