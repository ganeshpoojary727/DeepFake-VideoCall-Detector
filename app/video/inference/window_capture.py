"""Window capture and sliding frame window buffer module for real-time inference."""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class WindowCapture:
    """Sliding temporal frame window capture buffer for real-time video streams."""

    def __init__(self, window_size: int = 16, stride: int = 1) -> None:
        self.window_size = window_size
        self.stride = stride
        self._buffer: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """Add frame array to buffer and return frame window when full.

        Args:
            frame: RGB frame numpy array [H, W, 3].

        Returns:
            Optional[List[np.ndarray]]: Full window frame list or None if buffer building.
        """
        self._buffer.append(frame)
        if len(self._buffer) >= self.window_size:
            window = list(self._buffer[: self.window_size])
            self._buffer = self._buffer[self.stride :]
            return window
        return None

    def clear(self) -> None:
        """Clear window capture buffer."""
        self._buffer.clear()
