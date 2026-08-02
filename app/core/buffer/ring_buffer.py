"""Fixed-capacity circular ring buffer for rolling audio tensors and video frame sequences."""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple, Union
import numpy as np
import torch


class RingBuffer:
    """Thread-safe circular ring buffer for storing rolling audio samples and video frame arrays.

    Supports zero-copy slicing views and rolling window snapshots (default 20 seconds).
    """

    def __init__(
        self,
        capacity_samples_or_frames: int = 600,  # e.g., 20 sec @ 30 FPS = 600 frames or 20 sec @ 16kHz = 320k samples
        duration_seconds: float = 20.0,
        sample_rate_or_fps: float = 30.0,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.sample_rate_or_fps = sample_rate_or_fps
        self.capacity = capacity_samples_or_frames or int(duration_seconds * sample_rate_or_fps)

        self._buffer: List[Union[np.ndarray, torch.Tensor]] = []
        self._write_pos: int = 0
        self._size: int = 0
        self._lock = threading.Lock()

    def append(self, item: Union[np.ndarray, torch.Tensor]) -> None:
        """Append sample item (frame array or audio tensor) to circular buffer."""
        with self._lock:
            if len(self._buffer) < self.capacity:
                self._buffer.append(item)
            else:
                self._buffer[self._write_pos] = item

            self._write_pos = (self._write_pos + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1

    def append_batch(self, items: List[Union[np.ndarray, torch.Tensor]]) -> None:
        """Append batch sequence of items to buffer."""
        for item in items:
            self.append(item)

    def get_latest(self, count: Optional[int] = None) -> List[Union[np.ndarray, torch.Tensor]]:
        """Get ordered list of latest items in temporal sequence.

        Args:
            count: Optional max item count to retrieve.

        Returns:
            List[Union[np.ndarray, torch.Tensor]]: Ordered chronological list of items.
        """
        with self._lock:
            if self._size == 0:
                return []

            req_count = self._size if count is None else min(count, self._size)
            if self._size < self.capacity:
                items = self._buffer[self._size - req_count : self._size]
            else:
                # Circular reconstruction
                start_idx = (self._write_pos - req_count) % self.capacity
                if start_idx < self._write_pos:
                    items = self._buffer[start_idx:self._write_pos]
                else:
                    items = self._buffer[start_idx:] + self._buffer[:self._write_pos]
            return list(items)

    def get_as_tensor(self) -> Optional[torch.Tensor]:
        """Convert current buffer contents into a concatenated PyTorch Tensor."""
        items = self.get_latest()
        if not items:
            return None

        first = items[0]
        if isinstance(first, np.ndarray):
            arr_list = [t if isinstance(t, np.ndarray) else t.cpu().numpy() for t in items]
            stacked = np.stack(arr_list, axis=0) if arr_list[0].ndim >= 1 else np.array(arr_list)
            return torch.from_numpy(stacked)
        elif isinstance(first, torch.Tensor):
            return torch.stack(items, dim=0)
        return None

    def clear(self) -> None:
        """Reset buffer to empty state."""
        with self._lock:
            self._buffer.clear()
            self._write_pos = 0
            self._size = 0

    def __len__(self) -> int:
        """Get current size of buffer."""
        with self._lock:
            return self._size
