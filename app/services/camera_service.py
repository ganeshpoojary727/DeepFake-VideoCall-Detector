"""
Camera and Screen capture service using OpenCV and ScreenCapture.

Captures frames from webcam OR screen capture (for remote caller face analysis)
on a background thread and delivers them via `CameraFrameEvent` on the EventBus.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from app.config.settings import settings
from app.monitoring.screen_capture import ScreenCapture
from app.services.event_bus import CameraFrameEvent, ServiceStateEvent, event_bus
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CameraService:
    """
    Background webcam and screen capture service.

    Parameters
    ----------
    device_index : int
        OpenCV camera index (0 = default webcam).
    target_fps : int
        Target frame rate.
    width : int
        Requested width (pixels).
    height : int
        Requested height (pixels).
    video_source : str, optional
        "screen" (ScreenCapture for remote caller face) or "webcam" (local camera).
    """

    def __init__(
        self,
        device_index: int = 0,
        target_fps: int = 15,
        width: int = 640,
        height: int = 480,
        video_source: Optional[str] = None,
    ) -> None:
        self.device_index = device_index
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.video_source = video_source or getattr(settings.inference, "video_source", "screen")

        self._cap = None
        self._screen_cap: Optional[ScreenCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_frame(self):
        """Return the most recent captured frame (numpy ndarray BGR), or None."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def start(self) -> bool:
        """Start the capture thread (screen capture or webcam)."""
        if self._running:
            logger.warning("CameraService already running")
            return True

        self._stop_event.clear()

        # Enforce Screen Capture Mode (Remote Caller / Video Call Window)
        self.video_source = "screen"
        try:
            self._screen_cap = ScreenCapture(target_fps=self.target_fps)
            logger.info("CameraService initialized in Screen Capture mode (remote caller video calls)")
        except Exception as exc:
            logger.error("Screen capture init failed: %s", exc)
            return False

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraWorker",
            daemon=True,
        )
        self._thread.start()
        self._running = True

        event_bus.publish(ServiceStateEvent(service="CameraService", running=True))
        logger.info("CameraService started (source=%s, %dx%d @ %dfps)", self.video_source, self.width, self.height, self.target_fps)
        return True

    def stop(self) -> None:
        """Stop the capture thread and release resources."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._screen_cap is not None:
            self._screen_cap.close()
            self._screen_cap = None

        self._running = False
        event_bus.publish(ServiceStateEvent(service="CameraService", running=False))
        logger.info("CameraService stopped")

    def _capture_loop(self) -> None:
        """Capture loop for screen or webcam."""
        frame_interval = 1.0 / self.target_fps

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            frame = None

            if self.video_source == "screen" and self._screen_cap is not None:
                try:
                    frame = self._screen_cap.capture()
                except Exception as exc:
                    logger.debug("Screen capture frame error: %s", exc)

            elif self.video_source == "webcam" and self._cap is not None:
                ret, frame = self._cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            if frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                event_bus.publish(CameraFrameEvent(frame=frame))

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
