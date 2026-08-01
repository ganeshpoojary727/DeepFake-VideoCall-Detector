"""
Screen Capture Module — window-targeted capture with frame ring buffer.

Uses ``mss`` to capture only the active video call window region at 5 FPS.
Frames are stored in an in-memory ring buffer (``collections.deque``) for
15 seconds of rolling video context (75 frames). ZERO disk file writes.
"""

from __future__ import annotations

import collections
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ScreenCapture:
    """
    Captures screen regions and maintains a frame ring buffer.

    Uses ``mss`` for high-performance desktop grabbing with a ring buffer
    that holds 15 seconds of frames at the configured FPS.

    Parameters
    ----------
    target_fps : int
        Target frame rate for capture (default: 5).
    buffer_duration : float
        Ring buffer duration in seconds (default: 15.0).
    region : dict, optional
        Bounding box: ``{'top': y, 'left': x, 'width': w, 'height': h}``.
        If ``None``, captures the primary monitor.
    """

    def __init__(
        self,
        target_fps: Optional[int] = None,
        buffer_duration: float = 15.0,
        region: Optional[Dict[str, int]] = None,
    ) -> None:
        self.target_fps = target_fps or settings.VIDEO_TARGET_FPS
        self.buffer_duration = buffer_duration
        self.region = region
        self._sct = None

        # Frame ring buffer: 15s × 5 FPS = 75 frames
        max_frames = int(self.target_fps * buffer_duration)
        self._frame_buffer: collections.deque = collections.deque(maxlen=max_frames)

    def _init_mss(self) -> None:
        """Initialize mss instance lazily."""
        if self._sct is None:
            try:
                import mss

                # Support both mss.mss() and mss.MSS() for version compatibility
                if hasattr(mss, "MSS"):
                    self._sct = mss.MSS()
                else:
                    self._sct = mss.mss()
            except Exception as exc:
                logger.debug("Failed to initialize mss: %s", exc)

    def set_region(self, geometry: Tuple[int, int, int, int]) -> None:
        """
        Update the capture region from a window geometry tuple.

        Parameters
        ----------
        geometry : tuple[int, int, int, int]
            ``(left, top, width, height)`` from ``ProcessMonitor.get_window_geometry()``.
        """
        left, top, width, height = geometry
        self.region = {
            "top": max(0, top),
            "left": max(0, left),
            "width": max(1, width),
            "height": max(1, height),
        }
        logger.debug(
            "Screen capture region updated: left=%d, top=%d, w=%d, h=%d",
            left, top, width, height,
        )

    def capture(self) -> np.ndarray:
        """
        Capture a single frame from the screen or region.

        Returns
        -------
        np.ndarray
            BGR image array ``(H, W, 3)`` suitable for OpenCV processing.
        """
        # Strategy 1: High performance mss capture
        self._init_mss()
        if self._sct is not None:
            try:
                monitor = self.region or self._sct.monitors[1]
                sct_img = self._sct.grab(monitor)
                # Convert BGRA to BGR numpy array
                frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]
                return frame
            except Exception as exc:
                logger.debug("mss grab failed: %s; trying PIL fallback", exc)

        # Strategy 2: PIL ImageGrab fallback
        try:
            from PIL import ImageGrab

            bbox = (
                (
                    self.region["left"],
                    self.region["top"],
                    self.region["left"] + self.region["width"],
                    self.region["top"] + self.region["height"],
                )
                if self.region
                else None
            )
            pil_img = ImageGrab.grab(bbox=bbox)
            frame_rgb = np.array(pil_img, dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return frame_bgr
        except Exception as exc:
            logger.debug("PIL ImageGrab failed: %s; returning fallback frame", exc)

        # Strategy 3: Synthetic test frame if no display server attached
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Screen Capture Fallback",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        return blank

    def capture_to_buffer(self) -> np.ndarray:
        """
        Capture a single frame and append it to the ring buffer.

        Returns
        -------
        np.ndarray
            The captured BGR frame.
        """
        frame = self.capture()
        self._frame_buffer.append(frame)
        return frame

    def get_15s_buffer(self) -> List[np.ndarray]:
        """
        Return all frames currently in the ring buffer.

        Returns
        -------
        list[np.ndarray]
            List of up to 75 BGR frames (15s at 5 FPS).
        """
        return list(self._frame_buffer)

    def get_30s_buffer(self) -> List[np.ndarray]:
        """Backward compatibility alias for get_15s_buffer()."""
        return self.get_15s_buffer()

    def get_buffer_fill_ratio(self) -> float:
        """
        Return how full the frame ring buffer is (0.0 to 1.0).

        Useful for the warmup progress bar.
        """
        total = self._frame_buffer.maxlen or 1
        return min(len(self._frame_buffer) / total, 1.0)

    def get_frame_count(self) -> int:
        """Return the number of frames currently in the buffer."""
        return len(self._frame_buffer)

    def clear_buffer(self) -> None:
        """Clear the frame ring buffer."""
        self._frame_buffer.clear()

    def close(self) -> None:
        """Release underlying capture resources."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
