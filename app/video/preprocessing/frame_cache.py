"""Frame caching module for in-memory and disk caching of preprocessed video frames."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.video.constants.video_constants import VIDEO_CACHE_DIR

logger = logging.getLogger(__name__)


class FrameCache:
    """In-memory and disk cache for preprocessed frame arrays."""

    def __init__(self, cache_dir: str | Path = VIDEO_CACHE_DIR, max_memory_items: int = 100) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_memory_items = max_memory_items
        self._mem_cache: Dict[str, List[np.ndarray]] = {}
        self._mem_cache_arr: Dict[str, np.ndarray] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def compute_key(path: str | Path, config: Any) -> str:
        """Generate deterministic cache key based on video path, file stat, and preprocessing parameters."""
        path_str = str(Path(path).resolve())
        st_mtime = 0.0
        st_size = 0
        if os.path.exists(path_str):
            st = os.stat(path_str)
            st_mtime = st.st_mtime
            st_size = st.st_size

        seq_len = getattr(config, "sequence_length", 16)
        res = getattr(config, "target_resolution", (224, 224))
        crop = getattr(config, "crop_faces", True)
        strat = getattr(config, "sampling_strategy", "uniform")
        stride = getattr(config, "frame_stride", 1)

        key_data = f"{path_str}_{st_mtime}_{st_size}_{seq_len}_{res}_{crop}_{strat}_{stride}"
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def get_array(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached frame array from memory or disk cache."""
        if key in self._mem_cache_arr:
            return self._mem_cache_arr[key]

        disk_path = self.cache_dir / f"{key}.npz"
        if disk_path.exists():
            try:
                data = np.load(disk_path)["arr"]
                if len(self._mem_cache_arr) < self.max_memory_items:
                    self._mem_cache_arr[key] = data
                return data
            except Exception as exc:
                logger.warning("Failed to read frame cache disk file %s: %s", disk_path, exc)

        return None

    def put_array(self, key: str, arr: np.ndarray) -> None:
        """Store frame array in memory and disk cache."""
        if len(self._mem_cache_arr) >= self.max_memory_items:
            oldest_key = next(iter(self._mem_cache_arr))
            del self._mem_cache_arr[oldest_key]

        self._mem_cache_arr[key] = arr
        disk_path = self.cache_dir / f"{key}.npz"
        try:
            np.savez_compressed(disk_path, arr=arr)
        except Exception as exc:
            logger.warning("Failed to write frame cache disk file %s: %s", disk_path, exc)

    def get(self, key: str) -> Optional[List[np.ndarray]]:
        """Retrieve cached frame list by key identifier."""
        if key in self._mem_cache:
            return self._mem_cache[key]
        return None

    def put(self, key: str, frames: List[np.ndarray]) -> None:
        """Store frame list in cache under key identifier."""
        if len(self._mem_cache) >= self.max_memory_items:
            # Evict oldest key
            oldest_key = next(iter(self._mem_cache))
            del self._mem_cache[oldest_key]
        self._mem_cache[key] = frames

    def clear(self) -> None:
        """Clear memory cache."""
        self._mem_cache.clear()
        self._mem_cache_arr.clear()


# Alias for prompt requirement
FrameCaching = FrameCache

