"""
Monitoring service — connects ProcessMonitor with EventBus.

Scans running processes for video call applications (Zoom, Teams, Meet, WhatsApp, etc.)
and triggers auto-detection events when an active call is detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.config.settings import settings
from app.monitoring.process_monitor import ProcessInfo, ProcessMonitor
from app.services.event_bus import StatusEvent, VideoCallDetectedEvent, event_bus
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoCallStatus:
    """Status of detected video call applications."""

    is_active: bool
    detected_apps: List[str]
    process_list: List[ProcessInfo]

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.is_active:
            return "No video call detected"
        return f"Active: {', '.join(self.detected_apps)}"


class MonitoringService:
    """
    Service for video call process monitoring.

    Integrates `ProcessMonitor` with `EventBus` to provide real-time notifications
    and trigger auto-start detection when a call starts.
    """

    def __init__(self) -> None:
        self._proc_monitor = ProcessMonitor()
        self._is_monitoring = False

    def check_video_calls(self) -> VideoCallStatus:
        """Scan running processes once and return current status."""
        processes = self._proc_monitor.scan_processes()
        apps = sorted(list({p.display_name for p in processes}))
        return VideoCallStatus(
            is_active=len(apps) > 0,
            detected_apps=apps,
            process_list=processes,
        )

    def start_monitoring(self, interval_seconds: float = 3.0) -> None:
        """
        Start continuous background process monitoring.

        Events are automatically published on the EventBus.
        """
        if self._is_monitoring:
            return

        def _on_status_change(is_active: bool, active_procs: List[ProcessInfo]) -> None:
            apps = sorted(list({p.display_name for p in active_procs}))
            msg = f"Video call status: {'Active (' + ', '.join(apps) + ')' if is_active else 'Inactive'}"
            level = "warning" if is_active else "info"
            
            # Publish general status update
            event_bus.publish(StatusEvent(message=msg, level=level))

            # Publish VideoCallDetectedEvent for auto-starting detection pipeline
            event_bus.publish(
                VideoCallDetectedEvent(
                    is_active=is_active,
                    apps=apps,
                    auto_start=getattr(settings.inference, "auto_detect", True),
                )
            )

        self._proc_monitor.start_background_monitoring(
            on_status_change=_on_status_change,
            interval_seconds=interval_seconds,
        )
        self._is_monitoring = True
        logger.info("MonitoringService: background process scanning active")

    def stop_monitoring(self) -> None:
        """Stop background process monitoring."""
        if not self._is_monitoring:
            return
        self._proc_monitor.stop_background_monitoring()
        self._is_monitoring = False
        logger.info("MonitoringService: background process scanning stopped")
