"""System resource, FPS, latency, and queue size health monitor."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
import logging
import psutil
import threading
import time
from typing import Any, Dict, List, Optional
import torch

from app.core.queue.queue_manager import QueueManager

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Dataclass storing system metrics snapshot at a point in time."""

    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    fps: float
    avg_inference_latency_ms: float
    queue_sizes: Dict[str, int] = field(default_factory=dict)


class HealthMonitor:
    """Monitors system resource utilization, inference throughput, latency, and pipeline health."""

    def __init__(
        self,
        queue_manager: Optional[QueueManager] = None,
        latency_window_size: int = 50,
        fps_window_size: int = 50,
    ) -> None:
        self.queue_manager = queue_manager
        self._latencies_ms: collections.deque[float] = collections.deque(maxlen=latency_window_size)
        self._frame_timestamps: collections.deque[float] = collections.deque(maxlen=fps_window_size)
        self._lock = threading.Lock()

    def record_inference_latency(self, latency_ms: float) -> None:
        """Record model inference step duration in milliseconds."""
        with self._lock:
            self._latencies_ms.append(latency_ms)

    def record_frame(self) -> None:
        """Record a single frame timestamp for FPS calculation."""
        with self._lock:
            self._frame_timestamps.append(time.monotonic())

    def get_fps(self) -> float:
        """Calculate current processing FPS over recent frame window."""
        with self._lock:
            if len(self._frame_timestamps) < 2:
                return 0.0
            dt = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if dt <= 0:
                return 0.0
            return float((len(self._frame_timestamps) - 1) / dt)

    def get_average_latency_ms(self) -> float:
        """Calculate mean inference latency in milliseconds."""
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            return float(sum(self._latencies_ms) / len(self._latencies_ms))

    def collect_snapshot(self) -> HealthSnapshot:
        """Collect and return real-time system and runtime health metrics snapshot."""
        cpu_p = float(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        ram_p = float(mem.percent)
        ram_mb = float(mem.used / (1024 * 1024))

        gpu_alloc_mb = 0.0
        gpu_res_mb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_alloc_mb = float(torch.cuda.memory_allocated() / (1024 * 1024))
                gpu_res_mb = float(torch.cuda.memory_reserved() / (1024 * 1024))
            except Exception:
                pass

        q_sizes: Dict[str, int] = {}
        if self.queue_manager is not None:
            stats = self.queue_manager.get_all_stats()
            q_sizes = {name: s.size for name, s in stats.items()}

        return HealthSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_p,
            ram_percent=ram_p,
            ram_used_mb=ram_mb,
            gpu_allocated_mb=gpu_alloc_mb,
            gpu_reserved_mb=gpu_res_mb,
            fps=self.get_fps(),
            avg_inference_latency_ms=self.get_average_latency_ms(),
            queue_sizes=q_sizes,
        )
