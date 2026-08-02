"""Rolling 20-second temporary audio/video storage service preventing disk file accumulation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StoredSegment:
    """Dataclass holding stored segment metadata and location path or memory buffer."""

    segment_id: str
    modality: str
    timestamp: float
    duration_seconds: float
    filepath: Optional[Path] = None
    data: Optional[Any] = None


class TemporaryStorageService:
    """Manages rolling 20-second audio and video buffer files with auto-pruning."""

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        retention_seconds: float = 20.0,
        use_memory_backend: bool = False,
    ) -> None:
        self.retention_seconds = retention_seconds
        self.use_memory_backend = use_memory_backend
        self._lock = threading.Lock()
        self._segments: Dict[str, StoredSegment] = {}

        if not use_memory_backend:
            if storage_dir is not None:
                self.storage_dir = Path(storage_dir)
            else:
                self.storage_dir = Path(tempfile.gettempdir()) / "deepfake_detector_rolling_storage"
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_dir = Path(tempfile.gettempdir())

    def save_segment(
        self,
        segment_id: str,
        modality: str,
        data: Any,
        duration_seconds: float = 1.0,
    ) -> StoredSegment:
        """Save new audio/video segment and trigger immediate pruning of outdated files."""
        now = time.time()
        filepath: Optional[Path] = None

        with self._lock:
            if not self.use_memory_backend:
                ext = ".npy" if isinstance(data, np.ndarray) else ".bin"
                fname = f"{modality}_{segment_id}_{int(now*1000)}{ext}"
                filepath = self.storage_dir / fname

                if isinstance(data, np.ndarray):
                    np.save(filepath, data)
                elif isinstance(data, bytes):
                    filepath.write_bytes(data)
                else:
                    filepath.write_text(str(data))

            seg = StoredSegment(
                segment_id=segment_id,
                modality=modality,
                timestamp=now,
                duration_seconds=duration_seconds,
                filepath=filepath,
                data=data if self.use_memory_backend else None,
            )
            self._segments[segment_id] = seg
            self._prune_locked(now)
            return seg

    def get_segment(self, segment_id: str) -> Optional[StoredSegment]:
        """Retrieve stored segment metadata and content."""
        with self._lock:
            return self._segments.get(segment_id)

    def prune_old_segments(self) -> int:
        """Manually trigger pruning of segments older than retention threshold."""
        with self._lock:
            return self._prune_locked(time.time())

    def _prune_locked(self, current_time: float) -> int:
        """Internal helper to remove files and entries exceeding retention cutoff."""
        cutoff = current_time - self.retention_seconds
        to_delete = [
            sid for sid, seg in self._segments.items()
            if seg.timestamp < cutoff
        ]

        pruned_count = 0
        for sid in to_delete:
            seg = self._segments.pop(sid)
            if seg.filepath and seg.filepath.exists():
                try:
                    os.remove(seg.filepath)
                    pruned_count += 1
                except Exception as err:
                    logger.warning(f"Failed to remove temp segment file {seg.filepath}: {err}")
        return pruned_count

    def clear(self) -> None:
        """Purge all stored temporary files and metadata."""
        with self._lock:
            for seg in self._segments.values():
                if seg.filepath and seg.filepath.exists():
                    try:
                        os.remove(seg.filepath)
                    except Exception:
                        pass
            self._segments.clear()
            if not self.use_memory_backend and self.storage_dir.exists():
                try:
                    shutil.rmtree(self.storage_dir, ignore_errors=True)
                    self.storage_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
