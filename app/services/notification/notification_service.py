"""Runtime event notification and fake detection alert engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationSeverity(enum.Enum):
    """Notification severity level."""

    INFO = "INFO"
    WARNING = "WARNING"
    ALERT = "ALERT"
    CRITICAL = "CRITICAL"


@dataclass
class NotificationEvent:
    """Dataclass representing a runtime event or notification."""

    title: str
    message: str
    severity: NotificationSeverity
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


NotificationCallback = Callable[[NotificationEvent], None]


class NotificationService:
    """Publishes runtime events, system alerts, and deepfake detection notifications."""

    def __init__(self, fake_alert_threshold: float = 0.8) -> None:
        self.fake_alert_threshold = fake_alert_threshold
        self._subscribers: List[NotificationCallback] = []
        self._lock = threading.Lock()
        self._history: List[NotificationEvent] = []

    def subscribe(self, callback: NotificationCallback) -> None:
        """Register subscriber callback function for notifications."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: NotificationCallback) -> None:
        """Unregister subscriber callback function."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(
        self,
        title: str,
        message: str,
        severity: NotificationSeverity = NotificationSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotificationEvent:
        """Publish runtime notification event to all subscribers.

        Args:
            title: Short notification title.
            message: Descriptive message body.
            severity: NotificationSeverity level.
            metadata: Optional additional metadata dict.

        Returns:
            NotificationEvent: Emitted notification event instance.
        """
        event = NotificationEvent(
            title=title,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )

        with self._lock:
            self._history.append(event)
            subs = list(self._subscribers)

        log_fn = {
            NotificationSeverity.INFO: logger.info,
            NotificationSeverity.WARNING: logger.warning,
            NotificationSeverity.ALERT: logger.warning,
            NotificationSeverity.CRITICAL: logger.error,
        }.get(severity, logger.info)

        log_fn(f"[{severity.value}] {title}: {message}")

        for sub in subs:
            try:
                sub(event)
            except Exception as err:
                logger.error(f"Error executing notification subscriber callback: {err}")

        return event

    def notify_fake_detected(
        self,
        confidence: float,
        modality: str = "fused",
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[NotificationEvent]:
        """Trigger high-priority alert if fake detection confidence exceeds threshold."""
        if confidence >= self.fake_alert_threshold:
            title = "DEEPFAKE DETECTED"
            msg = f"Potential deepfake media detected on {modality} channel with {confidence*100.0:.1f}% confidence."
            meta = details or {}
            meta["confidence"] = confidence
            meta["modality"] = modality
            return self.publish(
                title=title,
                message=msg,
                severity=NotificationSeverity.ALERT,
                metadata=meta,
            )
        return None

    def get_history(self) -> List[NotificationEvent]:
        """Fetch copy of recent notification event history."""
        with self._lock:
            return list(self._history)
