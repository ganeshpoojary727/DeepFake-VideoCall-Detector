"""Thread supervisor engine for monitoring worker thread liveness, heartbeats, and auto-restart."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThreadInfo:
    """Dataclass holding registered worker thread status and heartbeat records."""

    name: str
    target: Callable[[], None]
    thread: Optional[threading.Thread] = None
    last_heartbeat: float = field(default_factory=time.monotonic)
    is_alive: bool = True
    restart_count: int = 0
    max_restarts: int = 5


class ThreadSupervisor:
    """Monitors worker threads, checks heartbeats, and restarts failed threads automatically."""

    def __init__(
        self,
        check_interval: float = 1.0,
        heartbeat_timeout: float = 5.0,
    ) -> None:
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._threads: Dict[str, ThreadInfo] = {}
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def register_thread(
        self,
        name: str,
        target: Callable[[], None],
        max_restarts: int = 5,
    ) -> ThreadInfo:
        """Register worker thread target with supervisor."""
        with self._lock:
            info = ThreadInfo(name=name, target=target, max_restarts=max_restarts)
            self._threads[name] = info
            return info

    def start_thread(self, name: str) -> bool:
        """Start a registered worker thread."""
        with self._lock:
            info = self._threads.get(name)
            if info is None:
                logger.error(f"Thread '{name}' not registered.")
                return False

            if info.thread is not None and info.thread.is_alive():
                return True

            t = threading.Thread(target=info.target, name=name, daemon=True)
            info.thread = t
            info.last_heartbeat = time.monotonic()
            info.is_alive = True
            t.start()
            logger.info(f"Started supervisor worker thread '{name}'.")
            return True

    def heartbeat(self, name: str) -> None:
        """Record heartbeat pulse for registered worker thread."""
        with self._lock:
            info = self._threads.get(name)
            if info is not None:
                info.last_heartbeat = time.monotonic()
                info.is_alive = True

    def start(self) -> None:
        """Start background supervisor monitoring loop."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, name="ThreadSupervisor", daemon=True)
            self._monitor_thread.start()
            logger.info("ThreadSupervisor monitoring started.")

    def stop(self) -> None:
        """Stop supervisor monitoring loop."""
        with self._lock:
            self._running = False
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("ThreadSupervisor stopped.")

    def _monitor_loop(self) -> None:
        """Internal monitoring loop verifying liveness and heartbeats."""
        while self._running:
            now = time.monotonic()
            with self._lock:
                for name, info in list(self._threads.items()):
                    if info.thread is None or not info.thread.is_alive():
                        info.is_alive = False
                        self._handle_thread_failure(info, reason="Thread terminated")
                    elif (now - info.last_heartbeat) > self.heartbeat_timeout:
                        info.is_alive = False
                        self._handle_thread_failure(info, reason="Heartbeat timeout")
            time.sleep(self.check_interval)

    def _handle_thread_failure(self, info: ThreadInfo, reason: str) -> None:
        """Handle worker thread restart procedure."""
        logger.warning(f"Worker thread '{info.name}' failed ({reason}). Restarts: {info.restart_count}/{info.max_restarts}")
        if info.restart_count < info.max_restarts:
            info.restart_count += 1
            t = threading.Thread(target=info.target, name=info.name, daemon=True)
            info.thread = t
            info.last_heartbeat = time.monotonic()
            info.is_alive = True
            t.start()
            logger.info(f"Restarted thread '{info.name}' (attempt {info.restart_count}).")
        else:
            logger.error(f"Thread '{info.name}' exceeded maximum restart threshold.")

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Fetch snapshot dictionary of all monitored thread statuses."""
        with self._lock:
            return {
                name: {
                    "is_alive": info.is_alive and (info.thread is not None and info.thread.is_alive()),
                    "restart_count": info.restart_count,
                    "last_heartbeat": info.last_heartbeat,
                }
                for name, info in self._threads.items()
            }
