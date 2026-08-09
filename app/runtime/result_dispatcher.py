"""
Result Dispatcher — synchronises audio and video predictions, triggers fusion.

Design
──────
• Runs its own background thread that drains audio and video result queues.
• Maintains the *latest* prediction for each modality with a timestamp.
• When both modalities have recent predictions (within ``sync_window_sec``),
  triggers ``MultimodalFusion.evaluate()`` for a fused decision.
• Falls back to single-modality results when only one stream is active.
• Publishes fused results via the existing ``EventBus``.

Thread Safety
─────────────
Multiple workers push results concurrently into their respective queues.
The dispatcher is the sole consumer of both result queues, so no locking
is needed on the queue reads.  The ``_latest_*`` fields are protected by
a lock for safe reads from ``get_latest_*()`` accessors.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from app.config.settings import settings
from app.core.interfaces import DetectionLabel, Modality, PredictionResult
from app.core.queue.queue_manager import BoundedQueue
from app.fusion.inference.fusion_engine import MultimodalFusion
from app.services.event_bus import DetectionEvent, event_bus
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResultDispatcher:
    """Synchronises audio/video predictions and dispatches fused results.

    Parameters
    ----------
    audio_result_queue : BoundedQueue
        Queue of ``PredictionResult`` from the AudioWorker.
    video_result_queue : BoundedQueue
        Queue of ``PredictionResult`` from the VideoWorker.
    fusion_engine : MultimodalFusion
        The multimodal fusion engine instance.
    sync_window_sec : float
        Maximum age (seconds) for two modality predictions to be
        considered concurrent and eligible for fusion.  Default: 5.0.
    stop_event : threading.Event | None
        External stop signal.
    """

    THREAD_NAME = "ResultDispatcher"

    def __init__(
        self,
        audio_result_queue: BoundedQueue,
        video_result_queue: BoundedQueue,
        fusion_engine: MultimodalFusion,
        sync_window_sec: float = 5.0,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._audio_queue = audio_result_queue
        self._video_queue = video_result_queue
        self._fusion = fusion_engine
        self._sync_window = sync_window_sec
        self._stop_event = stop_event or threading.Event()

        # Latest predictions (protected by lock)
        self._lock = threading.Lock()
        self._latest_audio: Optional[PredictionResult] = None
        self._latest_audio_time: float = 0.0
        self._latest_video: Optional[PredictionResult] = None
        self._latest_video_time: float = 0.0
        self._latest_fused: Optional[Dict[str, Any]] = None

        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Counters
        self._total_fusions = 0
        self._total_audio_only = 0
        self._total_video_only = 0

    # ── Properties ────────────────────────────

    @property
    def is_running(self) -> bool:
        """Whether the dispatcher thread is alive."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def latest_fused(self) -> Optional[Dict[str, Any]]:
        """Latest fused prediction result (thread-safe read)."""
        with self._lock:
            return self._latest_fused

    @property
    def total_fusions(self) -> int:
        """Total number of fused predictions produced."""
        return self._total_fusions

    # ── Lifecycle ─────────────────────────────

    def start(self) -> None:
        """Start the dispatcher thread."""
        if self.is_running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name=self.THREAD_NAME,
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info("ResultDispatcher started (sync_window=%.1fs)", self._sync_window)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the dispatcher thread gracefully."""
        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("ResultDispatcher thread did not join within %.1fs", timeout)
            self._thread = None

        logger.info(
            "ResultDispatcher stopped (fusions=%d, audio_only=%d, video_only=%d)",
            self._total_fusions, self._total_audio_only, self._total_video_only,
        )

    # ── Main Loop ─────────────────────────────

    def _dispatch_loop(self) -> None:
        """Drain result queues and trigger fusion when both modalities are available."""
        logger.debug("ResultDispatcher loop started")

        while not self._stop_event.is_set():
            updated = False

            # Drain audio results (non-blocking)
            audio_result = self._audio_queue.get(block=False)
            if audio_result is not None:
                with self._lock:
                    self._latest_audio = audio_result
                    self._latest_audio_time = time.monotonic()
                updated = True

            # Drain video results (non-blocking)
            video_result = self._video_queue.get(block=False)
            if video_result is not None:
                with self._lock:
                    self._latest_video = video_result
                    self._latest_video_time = time.monotonic()
                updated = True

            # Attempt fusion if anything was updated
            if updated:
                self._try_fusion()

            # Brief sleep to avoid busy-spinning when queues are empty
            if not updated:
                self._stop_event.wait(timeout=0.1)

        logger.debug("ResultDispatcher loop exited")

    def _try_fusion(self) -> None:
        """Attempt to fuse the latest audio and video predictions."""
        now = time.monotonic()

        with self._lock:
            audio = self._latest_audio
            audio_age = now - self._latest_audio_time if audio else float("inf")
            video = self._latest_video
            video_age = now - self._latest_video_time if video else float("inf")

        audio_fresh = audio is not None and audio_age <= self._sync_window
        video_fresh = video is not None and video_age <= self._sync_window

        if audio_fresh and video_fresh:
            # Both modalities available — run fusion
            self._emit_fused(audio, video)
            self._total_fusions += 1

        elif audio_fresh and not video_fresh:
            # Audio only — publish audio-only result
            self._emit_single(audio)
            self._total_audio_only += 1

        elif video_fresh and not audio_fresh:
            # Video only — publish video-only result
            self._emit_single(video)
            self._total_video_only += 1

    def _emit_fused(self, audio: PredictionResult, video: PredictionResult) -> None:
        """Run MultimodalFusion and publish the fused result."""
        try:
            fusion_result = self._fusion.evaluate(audio.confidence, video.confidence)

            # Build a fused PredictionResult for the EventBus
            combined_score = fusion_result["combined_score"]

            if combined_score >= settings.inference.confidence_threshold_fake:
                label = DetectionLabel.FAKE
            elif combined_score <= settings.inference.confidence_threshold_real:
                label = DetectionLabel.REAL
            else:
                label = DetectionLabel.UNCERTAIN

            fused_prediction = PredictionResult(
                label=label,
                confidence=combined_score,
                modality=Modality.FUSED,
                latency_ms=round(audio.latency_ms + video.latency_ms, 2),
                model_version=settings.model.model_version,
            )

            # Store for status API
            with self._lock:
                self._latest_fused = fusion_result

            # Publish on EventBus
            event_bus.publish(DetectionEvent(result=fused_prediction))

            logger.info(
                "Fused prediction: %s (combined=%.4f, audio=%.4f, video=%.4f)",
                label.value, combined_score, audio.confidence, video.confidence,
            )

        except Exception as exc:
            logger.error("Fusion error: %s", exc)

    def _emit_single(self, result: PredictionResult) -> None:
        """Publish a single-modality result on the EventBus."""
        event_bus.publish(DetectionEvent(result=result))

        with self._lock:
            self._latest_fused = {
                "combined_score": result.confidence,
                "prediction": result.label.value,
                "modality": result.modality.value,
            }

        logger.debug(
            "Single-modality result: %s (modality=%s, conf=%.4f)",
            result.label.value, result.modality.value, result.confidence,
        )
