"""Session orchestrator for runtime service lifecycle management and coordination."""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Any, Dict, Optional

from app.core.queue.queue_manager import QueueManager
from app.monitoring.health.health_monitor import HealthMonitor
from app.monitoring.supervisor.thread_supervisor import ThreadSupervisor
from app.services.history.history_service import HistoryService
from app.services.notification.notification_service import NotificationService
from app.services.storage.temporary_storage_service import TemporaryStorageService

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    """Session lifecycle state."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class SessionOrchestrator:
    """Coordinates lifecycle, thread supervisor, health monitoring, and runtime services."""

    def __init__(
        self,
        queue_manager: Optional[QueueManager] = None,
        supervisor: Optional[ThreadSupervisor] = None,
        health_monitor: Optional[HealthMonitor] = None,
        storage_service: Optional[TemporaryStorageService] = None,
        history_service: Optional[HistoryService] = None,
        notification_service: Optional[NotificationService] = None,
    ) -> None:
        self.queue_manager = queue_manager or QueueManager()
        self.supervisor = supervisor or ThreadSupervisor()
        self.health_monitor = health_monitor or HealthMonitor(queue_manager=self.queue_manager)
        self.storage_service = storage_service or TemporaryStorageService(use_memory_backend=True)
        self.history_service = history_service or HistoryService()
        self.notification_service = notification_service or NotificationService()

        self._state = SessionState.IDLE
        self._lock = threading.Lock()
        self._session_start_time: Optional[float] = None

    @property
    def state(self) -> SessionState:
        """Current session lifecycle state."""
        with self._lock:
            return self._state

    def start(self) -> bool:
        """Start runtime session services."""
        with self._lock:
            if self._state == SessionState.RUNNING:
                return True

            logger.info("Starting SessionOrchestrator runtime services...")
            self.supervisor.start()
            self._state = SessionState.RUNNING
            self._session_start_time = time.time()

            self.notification_service.publish(
                title="Session Started",
                message="Runtime session orchestrator successfully initialized.",
            )
            return True

    def pause(self) -> bool:
        """Pause active runtime session."""
        with self._lock:
            if self._state != SessionState.RUNNING:
                return False
            self._state = SessionState.PAUSED
            self.notification_service.publish(title="Session Paused", message="Runtime session paused.")
            return True

    def resume(self) -> bool:
        """Resume paused runtime session."""
        with self._lock:
            if self._state != SessionState.PAUSED:
                return False
            self._state = SessionState.RUNNING
            self.notification_service.publish(title="Session Resumed", message="Runtime session resumed.")
            return True

    def stop(self) -> bool:
        """Gracefully stop runtime session and services."""
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.STOPPED):
                return True

            logger.info("Stopping SessionOrchestrator runtime services...")
            self._state = SessionState.STOPPED
            self.supervisor.stop()
            self.storage_service.clear()

            self.notification_service.publish(
                title="Session Stopped",
                message="Runtime session orchestrator stopped and cleaned up.",
            )
            return True

    def restart(self) -> bool:
        """Restart runtime session services."""
        self.stop()
        time.sleep(0.1)
        return self.start()

    def get_summary(self) -> Dict[str, Any]:
        """Fetch summary of session state, health snapshot, and statistics."""
        with self._lock:
            curr_state = self._state.value
            start_t = self._session_start_time

        uptime = (time.time() - start_t) if start_t and curr_state == "RUNNING" else 0.0
        health = self.health_monitor.collect_snapshot()

        return {
            "state": curr_state,
            "uptime_seconds": float(uptime),
            "thread_status": self.supervisor.get_status(),
            "health_snapshot": {
                "cpu_percent": health.cpu_percent,
                "ram_percent": health.ram_percent,
                "fps": health.fps,
                "avg_inference_latency_ms": health.avg_inference_latency_ms,
            },
            "history_record_count": len(self.history_service),
        }
