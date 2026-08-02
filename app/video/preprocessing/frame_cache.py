"""Frame caching module for in-memory and disk caching of preprocessed video frames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from app.video.constants.video_constants import VIDEO_CACHE_DIR


class FrameCache:
    """In-memory and disk cache for preprocessed frame arrays."""

    def __init__(self, cache_dir: str | Path = VIDEO_CACHE_DIR, max_memory_items: int = 100) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_memory_items = max_memory_items
        self._mem_cache: Dict[str, List[np.ndarray]] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

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


# Alias for prompt requirement
FrameCaching = FrameCache
