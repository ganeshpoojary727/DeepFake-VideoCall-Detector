"""
Thread-safe publish/subscribe event bus.

Decouples the AI detection pipeline from the GUI and other consumers.
Any component can publish events; any component can subscribe to event types.

Usage
-----
    from app.services.event_bus import event_bus, DetectionEvent

    # Subscribe
    event_bus.subscribe(DetectionEvent, my_callback)

    # Publish (from any thread)
    event_bus.publish(DetectionEvent(result=result))

    # Drain in the GUI thread via a QTimer polling loop
    events = event_bus.drain()
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from app.core.interfaces import PredictionResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Event types
# ──────────────────────────────────────────────


@dataclass
class BaseEvent:
    """Base class for all events published on the bus."""
    pass


@dataclass
class DetectionEvent(BaseEvent):
    """Fired when a detection result is available."""
    result: PredictionResult


@dataclass
class StatusEvent(BaseEvent):
    """Fired when the monitoring status changes (e.g. video call detected)."""
    message: str
    level: str = "info"  # "info" | "warning" | "error"


@dataclass
class VideoCallDetectedEvent(BaseEvent):
    """Fired when active video call status changes."""
    is_active: bool
    apps: List[str]
    auto_start: bool = True


@dataclass
class AudioLevelEvent(BaseEvent):
    """Fired periodically with the current RMS audio level (0.0–1.0)."""
    level: float


@dataclass
class CameraFrameEvent(BaseEvent):
    """Fired when a new camera frame is available."""
    frame: Any  # numpy ndarray (H, W, 3) BGR


@dataclass
class ServiceStateEvent(BaseEvent):
    """Fired when a service starts or stops."""
    service: str
    running: bool


# ──────────────────────────────────────────────
# Event Bus
# ──────────────────────────────────────────────


class EventBus:
    """
    Thread-safe, type-routed publish/subscribe event bus.

    Internally uses a ``queue.Queue`` so that events produced by background
    threads can be safely consumed by the GUI (main) thread.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: queue.Queue[BaseEvent] = queue.Queue(maxsize=maxsize)
        self._subscribers: Dict[Type[BaseEvent], List[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()

    # ── Subscription ──────────────────────────

    def subscribe(
        self,
        event_type: Type[BaseEvent],
        callback: Callable[[Any], None],
    ) -> None:
        """
        Register a callback for a specific event type.

        The callback is invoked from whichever thread calls ``drain()``
        or ``dispatch_all()``, so it must be the GUI thread for Qt usage.

        Parameters
        ----------
        event_type : Type[BaseEvent]
            The event class to subscribe to.
        callback : callable
            Function that receives the event instance.
        """
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug("Subscribed %s to %s", callback.__name__, event_type.__name__)

    def unsubscribe(
        self,
        event_type: Type[BaseEvent],
        callback: Callable[[Any], None],
    ) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            listeners = self._subscribers.get(event_type, [])
            if callback in listeners:
                listeners.remove(callback)

    # ── Publishing ────────────────────────────

    def publish(self, event: BaseEvent) -> None:
        """
        Publish an event from any thread.

        If the internal queue is full, the oldest event is discarded to
        prevent producer threads from blocking the AI pipeline.

        Parameters
        ----------
        event : BaseEvent
            The event to publish.
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            self._queue.put_nowait(event)
            logger.warning("EventBus queue full — dropped oldest event")

    # ── Consuming ─────────────────────────────

    def drain(self, max_events: int = 50) -> List[BaseEvent]:
        """
        Collect and return up to *max_events* queued events without blocking.

        Call this from a QTimer in the GUI thread.

        Returns
        -------
        List[BaseEvent]
            Events collected from the queue.
        """
        events: List[BaseEvent] = []
        for _ in range(max_events):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def dispatch_all(self, max_events: int = 50) -> None:
        """
        Drain the queue and invoke registered callbacks for each event.

        Must be called from the same thread as the subscribers (GUI thread).
        """
        for event in self.drain(max_events):
            event_type = type(event)
            with self._lock:
                callbacks = list(self._subscribers.get(event_type, []))
            for cb in callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    logger.error(
                        "Error in EventBus callback %s: %s", cb.__name__, exc
                    )

    def clear(self) -> None:
        """Discard all pending events."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ──────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────

event_bus = EventBus()
"""Module-level singleton event bus shared across the application."""
