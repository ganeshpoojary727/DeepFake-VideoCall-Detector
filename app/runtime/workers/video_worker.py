"""
Video inference worker — background thread for real-time video deepfake detection.

Pulls frame batches from a ``BoundedQueue``, runs
``VideoDeepfakeDetector.predict_from_frames()``, and pushes
``PredictionResult`` objects onto a result queue.

Threading Model
───────────────
• One long-lived daemon thread per VideoWorker instance.
• Uses ``threading.Event`` for cooperative shutdown.
• Sends heartbeats to ``ThreadSupervisor`` for liveness monitoring.
• Records inference latency in ``HealthMonitor`` for dashboard metrics.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np

from app.config.settings import settings
from app.core.interfaces import DetectionLabel, Modality, PredictionResult
from app.core.queue.queue_manager import BoundedQueue
from app.monitoring.health.health_monitor import HealthMonitor
from app.monitoring.supervisor.thread_supervisor import ThreadSupervisor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoWorker:
    """Background thread that continuously runs video deepfake inference.

    Parameters
    ----------
    video_detector : object
        An initialised ``VideoDeepfakeDetector`` (or any object with a
        ``predict_from_frames(List[np.ndarray]) -> float`` method).
    input_queue : BoundedQueue
        Queue of ``List[np.ndarray]`` frame batches to process.
    result_queue : BoundedQueue
        Queue to push ``PredictionResult`` objects onto.
    supervisor : ThreadSupervisor | None
        Optional supervisor for heartbeat reporting and auto-restart.
    health_monitor : HealthMonitor | None
        Optional health monitor for latency recording.
    stop_event : threading.Event | None
        External stop signal.  If ``None``, an internal event is created.
    """

    WORKER_NAME = "VideoWorker"

    def __init__(
        self,
        video_detector: object,
        input_queue: BoundedQueue,
        result_queue: BoundedQueue,
        supervisor: Optional[ThreadSupervisor] = None,
        health_monitor: Optional[HealthMonitor] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._detector = video_detector
        self._input_queue = input_queue
        self._result_queue = result_queue
        self._supervisor = supervisor
        self._health_monitor = health_monitor
        self._stop_event = stop_event or threading.Event()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._total_inferences = 0
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        with self._lock:
            return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def total_inferences(self) -> int:
        """Total number of inference runs completed."""
        with self._lock:
            return self._total_inferences

    @property
    def last_error(self) -> Optional[str]:
        """Last error message, if any."""
        with self._lock:
            return self._last_error

    # ── Lifecycle ─────────────────────────────

    def start(self) -> None:
        """Start the inference loop in a background thread."""
        if self.is_running:
            logger.debug("VideoWorker already running — skipping start")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name=self.WORKER_NAME,
            daemon=True,
        )
        self._thread.start()

        with self._lock:
            self._running = True

        logger.info("VideoWorker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for the thread to join."""
        self._stop_event.set()

        with self._lock:
            self._running = False

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("VideoWorker thread did not join within %.1fs", timeout)
            self._thread = None

        logger.info("VideoWorker stopped (total_inferences=%d)", self._total_inferences)

    # ── Inference Loop ────────────────────────

    def _inference_loop(self) -> None:
        """Main loop: pull frames → infer → push result → heartbeat."""
        logger.debug("VideoWorker inference loop started")

        while not self._stop_event.is_set():
            # Send heartbeat
            if self._supervisor is not None:
                self._supervisor.heartbeat(self.WORKER_NAME)

            # Pull frame batch from input queue (blocking with 1s timeout)
            frames_batch = self._input_queue.get(block=True, timeout=1.0)
            if frames_batch is None:
                # Timeout — no data, loop back to check stop event
                continue

            try:
                result = self._run_single_inference(frames_batch)
                self._result_queue.put(result)

                with self._lock:
                    self._total_inferences += 1

                # Record frame for FPS calculation
                if self._health_monitor is not None:
                    self._health_monitor.record_frame()

            except Exception as exc:
                error_msg = f"VideoWorker inference error: {exc}"
                logger.warning(error_msg)
                with self._lock:
                    self._last_error = error_msg

        logger.debug("VideoWorker inference loop exited")

    def _run_single_inference(self, frames: List[np.ndarray]) -> PredictionResult:
        """Execute a single video inference and return a PredictionResult."""
        start = time.perf_counter()

        # Delegate to the existing VideoDeepfakeDetector
        fake_probability = self._detector.predict_from_frames(frames)

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Record latency in health monitor
        if self._health_monitor is not None:
            self._health_monitor.record_inference_latency(latency_ms)

        # Three-way decision
        if fake_probability >= settings.inference.confidence_threshold_fake:
            label = DetectionLabel.FAKE
        elif fake_probability <= settings.inference.confidence_threshold_real:
            label = DetectionLabel.REAL
        else:
            label = DetectionLabel.UNCERTAIN

        result = PredictionResult(
            label=label,
            confidence=fake_probability,
            modality=Modality.VIDEO,
            latency_ms=round(latency_ms, 2),
            model_version=settings.model.model_version,
        )

        logger.debug(
            "VideoWorker inference: %s (conf=%.4f, frames=%d, latency=%.1fms)",
            result.label.value, result.confidence, len(frames), result.latency_ms,
        )

        return result

    # ── Target for ThreadSupervisor ───────────

    def get_loop_target(self):
        """Return a callable suitable for ``ThreadSupervisor.register_thread()``."""
        return self._inference_loop
