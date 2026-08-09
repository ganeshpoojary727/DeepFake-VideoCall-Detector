"""
Runtime status types — state enum and structured dataclasses.

These types form the public status API of the Runtime Orchestration Layer.
They have NO dependency on PyQt6, torch, or any AI module — they are pure
domain types that the future GUI (or any other consumer) can import cheaply.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# State Machine
# ──────────────────────────────────────────────


class RuntimeState(enum.Enum):
    """Runtime controller lifecycle states.

    State transitions::

        UNINITIALIZED → INITIALIZING → READY → RUNNING → STOPPING → STOPPED
                              ↓                    ↓
                            ERROR ←──────────── ERROR
    """

    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ──────────────────────────────────────────────
# Worker Status
# ──────────────────────────────────────────────


@dataclass
class WorkerStatus:
    """Status snapshot for a single inference worker thread."""

    name: str
    alive: bool = False
    heartbeat_age_seconds: float = 0.0
    restart_count: int = 0
    total_inferences: int = 0
    last_error: Optional[str] = None

    @property
    def healthy(self) -> bool:
        """Worker is alive and heartbeat is recent (< 10s)."""
        return self.alive and self.heartbeat_age_seconds < 10.0


# ──────────────────────────────────────────────
# Model Status
# ──────────────────────────────────────────────


@dataclass
class ModelStatus:
    """Status snapshot for a loaded AI model."""

    name: str
    loaded: bool = False
    device: str = "unknown"
    version: str = "unknown"
    load_time_ms: float = 0.0
    error: Optional[str] = None


# ──────────────────────────────────────────────
# Stream Status
# ──────────────────────────────────────────────


@dataclass
class StreamStatus:
    """Status snapshot for an active input stream (audio or video)."""

    name: str
    active: bool = False
    queue_size: int = 0
    queue_capacity: int = 0
    total_items_processed: int = 0
    drop_count: int = 0


# ──────────────────────────────────────────────
# Health Snapshot (lightweight mirror)
# ──────────────────────────────────────────────


@dataclass
class HealthSummary:
    """Lightweight health summary for the status API.

    Mirrors key fields from ``HealthSnapshot`` without requiring
    a direct dependency on the monitoring module.
    """

    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_mb: float = 0.0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    fps: float = 0.0
    avg_inference_latency_ms: float = 0.0


# ──────────────────────────────────────────────
# Top-Level Runtime Status
# ──────────────────────────────────────────────


@dataclass
class RuntimeStatus:
    """Complete structured status of the Runtime Orchestration Layer.

    Returned by ``RuntimeController.get_status()`` for GUI or CLI
    consumption. All fields are serialisation-friendly (no torch tensors,
    no threading objects).
    """

    state: RuntimeState = RuntimeState.UNINITIALIZED
    uptime_seconds: float = 0.0
    workers: List[WorkerStatus] = field(default_factory=list)
    models: List[ModelStatus] = field(default_factory=list)
    streams: List[StreamStatus] = field(default_factory=list)
    health: HealthSummary = field(default_factory=HealthSummary)
    last_fused_prediction: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire status to a plain dictionary."""
        return {
            "state": self.state.value,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "workers": [
                {
                    "name": w.name,
                    "alive": w.alive,
                    "healthy": w.healthy,
                    "heartbeat_age_seconds": round(w.heartbeat_age_seconds, 2),
                    "restart_count": w.restart_count,
                    "total_inferences": w.total_inferences,
                    "last_error": w.last_error,
                }
                for w in self.workers
            ],
            "models": [
                {
                    "name": m.name,
                    "loaded": m.loaded,
                    "device": m.device,
                    "version": m.version,
                    "load_time_ms": round(m.load_time_ms, 2),
                    "error": m.error,
                }
                for m in self.models
            ],
            "streams": [
                {
                    "name": s.name,
                    "active": s.active,
                    "queue_size": s.queue_size,
                    "queue_capacity": s.queue_capacity,
                    "total_items_processed": s.total_items_processed,
                    "drop_count": s.drop_count,
                }
                for s in self.streams
            ],
            "health": {
                "cpu_percent": self.health.cpu_percent,
                "ram_percent": self.health.ram_percent,
                "ram_used_mb": round(self.health.ram_used_mb, 1),
                "gpu_allocated_mb": round(self.health.gpu_allocated_mb, 1),
                "gpu_reserved_mb": round(self.health.gpu_reserved_mb, 1),
                "fps": round(self.health.fps, 1),
                "avg_inference_latency_ms": round(self.health.avg_inference_latency_ms, 2),
            },
            "last_fused_prediction": self.last_fused_prediction,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }
