"""Thread-safe frame queue module for real-time video streaming input."""

from __future__ import annotations

import queue
import threading
from typing import List, Optional
import numpy as np


class FrameQueue:
    """Thread-safe queue for streaming video frame frames into inference pipeline."""

    def __init__(self, maxsize: int = 128) -> None:
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Enqueue video frame array."""
        try:
            self._queue.put(frame, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Dequeue video frame array."""
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_batch(self, batch_size: int) -> List[np.ndarray]:
        """Dequeue batch of frames up to batch_size."""
        frames: List[np.ndarray] = []
        for _ in range(batch_size):
            frame = self.get(block=False)
            if frame is None:
                break
            frames.append(frame)
        return frames

    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def clear(self) -> None:
        """Drain frame queue."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
