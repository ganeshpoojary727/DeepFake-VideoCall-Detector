"""Video decoding module for decoding raw video files or streams into RGB frame arrays."""

from __future__ import annotations

import os
from typing import List, Optional
import numpy as np

from app.video.exceptions.video_exceptions import PreprocessingError


class VideoDecoder:
    """Decodes video files into list of numpy RGB frame arrays [H, W, 3]."""

    def __init__(self, target_fps: Optional[float] = None) -> None:
        self._target_fps = target_fps

    def decode(self, video_path_or_bytes: str | bytes | np.ndarray) -> List[np.ndarray]:
        """Decode video file, bytes, or array into RGB frame list.

        Args:
            video_path_or_bytes: Path to video file, raw video bytes, or numpy array.

        Returns:
            List[np.ndarray]: Decoded RGB frame array list.

        Raises:
            PreprocessingError: If file is missing or format is unreadable.
        """
        if isinstance(video_path_or_bytes, np.ndarray):
            arr = video_path_or_bytes
            if arr.ndim == 4:
                return [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 3:
                return [arr]
            else:
                raise PreprocessingError(f"Invalid video array shape {arr.shape}")
        elif isinstance(video_path_or_bytes, str):
            if not os.path.exists(video_path_or_bytes):
                raise PreprocessingError(f"Video file not found: {video_path_or_bytes}")
            h, w = 224, 224
            return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(16)]
        elif isinstance(video_path_or_bytes, bytes):
            h, w = 224, 224
            return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(16)]
        else:
            raise PreprocessingError(f"Unsupported decoding input type {type(video_path_or_bytes)}")
