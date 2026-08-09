"""
Runtime Controller — central orchestrator for the DeepFake Video Call Detector.

Coordinates the full application lifecycle without any GUI dependency:

1. Configuration loading and validation
2. Safe CUDA initialisation
3. Audio and video model loading (with graceful degradation)
4. Fusion engine setup
5. Worker thread lifecycle (start, monitor, stop)
6. Prediction synchronisation and fusion dispatch
7. Health monitoring and failure recovery
8. Structured runtime status API for the future GUI

Architecture
────────────
::

    ┌───────────────────────────────────────────────┐
    │               RuntimeController               │
    │                                               │
    │  AudioWorker ──► audio_results ──┐            │
    │                                  ├─► ResultDispatcher ──► EventBus
    │  VideoWorker ──► video_results ──┘            │
    │                                               │
    │  ThreadSupervisor ←── heartbeats              │
    │  HealthMonitor    ←── latencies               │
    │  QueueManager     ←── all queues              │
    └───────────────────────────────────────────────┘

State Machine
─────────────
::

    UNINITIALIZED → INITIALIZING → READY → RUNNING → STOPPING → STOPPED
                         ↓                    ↓
                       ERROR ←──────────── ERROR

Thread Safety
─────────────
• State transitions are protected by a reentrant lock.
• All cross-thread communication goes through ``BoundedQueue``.
• Worker threads are daemon threads; they do not prevent shutdown.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from app.config.settings import Settings, settings
from app.core.queue.queue_manager import BoundedQueue, QueueManager
from app.fusion.inference.fusion_engine import MultimodalFusion
from app.monitoring.health.health_monitor import HealthMonitor
from app.monitoring.supervisor.thread_supervisor import ThreadSupervisor
from app.runtime.result_dispatcher import ResultDispatcher
from app.runtime.runtime_status import (
    HealthSummary,
    ModelStatus,
    RuntimeState,
    RuntimeStatus,
    StreamStatus,
    WorkerStatus,
)
from app.runtime.workers.audio_worker import AudioWorker
from app.runtime.workers.video_worker import VideoWorker
from app.services.event_bus import ServiceStateEvent, StatusEvent, event_bus
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Valid state transitions
# ──────────────────────────────────────────────

_VALID_TRANSITIONS = {
    RuntimeState.UNINITIALIZED: {RuntimeState.INITIALIZING, RuntimeState.ERROR},
    RuntimeState.INITIALIZING: {RuntimeState.READY, RuntimeState.ERROR},
    RuntimeState.READY: {RuntimeState.RUNNING, RuntimeState.STOPPED, RuntimeState.ERROR},
    RuntimeState.RUNNING: {RuntimeState.STOPPING, RuntimeState.ERROR},
    RuntimeState.STOPPING: {RuntimeState.STOPPED, RuntimeState.ERROR},
    RuntimeState.STOPPED: {RuntimeState.INITIALIZING, RuntimeState.ERROR},
    RuntimeState.ERROR: {RuntimeState.INITIALIZING, RuntimeState.STOPPED},
}


class RuntimeController:
    """Central orchestrator for the DeepFake Video Call Detector runtime.

    Parameters
    ----------
    config : Settings | None
        Application configuration.  Defaults to the module-level singleton.
    sync_window_sec : float
        Time window for pairing audio/video results in the dispatcher.
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        sync_window_sec: float = 5.0,
    ) -> None:
        self._config = config or settings
        self._sync_window = sync_window_sec
        self._state = RuntimeState.UNINITIALIZED
        self._lock = threading.RLock()
        self._error_message: Optional[str] = None
        self._start_time: Optional[float] = None
        self._stop_event = threading.Event()

        # Components (initialised in ``initialize()``)
        self._queue_manager: Optional[QueueManager] = None
        self._audio_input_queue: Optional[BoundedQueue] = None
        self._video_input_queue: Optional[BoundedQueue] = None
        self._audio_result_queue: Optional[BoundedQueue] = None
        self._video_result_queue: Optional[BoundedQueue] = None

        self._voice_detector: Optional[Any] = None
        self._video_detector: Optional[Any] = None
        self._fusion_engine: Optional[MultimodalFusion] = None

        self._audio_worker: Optional[AudioWorker] = None
        self._video_worker: Optional[VideoWorker] = None
        self._result_dispatcher: Optional[ResultDispatcher] = None

        self._supervisor: Optional[ThreadSupervisor] = None
        self._health_monitor: Optional[HealthMonitor] = None

        # Model status tracking
        self._audio_model_status = ModelStatus(name="AASIST")
        self._video_model_status = ModelStatus(name="VideoDeepfakeDetector")

    # ══════════════════════════════════════════
    # STATE MACHINE
    # ══════════════════════════════════════════

    @property
    def state(self) -> RuntimeState:
        """Current runtime state (thread-safe read)."""
        with self._lock:
            return self._state

    def _set_state(self, new_state: RuntimeState) -> None:
        """Transition to a new state with validation."""
        with self._lock:
            if new_state not in _VALID_TRANSITIONS.get(self._state, set()):
                raise RuntimeError(
                    f"Invalid state transition: {self._state.value} → {new_state.value}"
                )
            old = self._state
            self._state = new_state

        logger.info("Runtime state: %s → %s", old.value, new_state.value)
        event_bus.publish(StatusEvent(
            message=f"Runtime state: {new_state.value}",
            level="info",
        ))

    # ══════════════════════════════════════════
    # INITIALIZATION
    # ══════════════════════════════════════════

    def initialize(self) -> bool:
        """Initialise all runtime components.

        Returns ``True`` on success, ``False`` on failure (state → ERROR).
        After successful initialisation, state becomes READY.
        """
        try:
            self._set_state(RuntimeState.INITIALIZING)

            # 1. Validate configuration
            warnings = self._config.validate()
            for w in warnings:
                logger.warning("Config warning: %s", w)

            # 2. Safe CUDA initialisation
            self._init_cuda()

            # 3. Create queue infrastructure
            self._init_queues()

            # 4. Load models (graceful degradation)
            self._load_audio_model()
            self._load_video_model()

            # 5. Initialise fusion engine
            self._fusion_engine = MultimodalFusion(
                audio_weight=self._config.AUDIO_WEIGHT,
                video_weight=self._config.VIDEO_WEIGHT,
            )
            logger.info(
                "Fusion engine ready (audio_w=%.2f, video_w=%.2f)",
                self._config.AUDIO_WEIGHT, self._config.VIDEO_WEIGHT,
            )

            # 6. Health monitoring + thread supervision
            self._health_monitor = HealthMonitor(queue_manager=self._queue_manager)
            self._supervisor = ThreadSupervisor(
                check_interval=2.0,
                heartbeat_timeout=10.0,
            )

            # 7. Create workers
            self._create_workers()

            # 8. Create result dispatcher
            self._result_dispatcher = ResultDispatcher(
                audio_result_queue=self._audio_result_queue,
                video_result_queue=self._video_result_queue,
                fusion_engine=self._fusion_engine,
                sync_window_sec=self._sync_window,
                stop_event=self._stop_event,
            )

            self._set_state(RuntimeState.READY)
            logger.info("Runtime initialization complete — state is READY")
            return True

        except Exception as exc:
            self._error_message = f"Initialization failed: {exc}"
            logger.error(self._error_message, exc_info=True)
            try:
                self._set_state(RuntimeState.ERROR)
            except RuntimeError:
                self._state = RuntimeState.ERROR
            return False

    def _init_cuda(self) -> None:
        """Safely probe CUDA availability without crashing."""
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                logger.info("CUDA available: %s (%.0f MB VRAM)", device_name, vram_mb)
            else:
                logger.info("CUDA not available — using CPU")
        except Exception as exc:
            logger.warning("CUDA probe failed (non-fatal): %s", exc)

    def _init_queues(self) -> None:
        """Create the bounded queues for inter-thread communication."""
        self._queue_manager = QueueManager()

        self._audio_input_queue = self._queue_manager.create_queue(
            "audio_input", maxsize=10, drop_oldest=True,
        )
        self._video_input_queue = self._queue_manager.create_queue(
            "video_input", maxsize=10, drop_oldest=True,
        )
        self._audio_result_queue = self._queue_manager.create_queue(
            "audio_results", maxsize=20, drop_oldest=True,
        )
        self._video_result_queue = self._queue_manager.create_queue(
            "video_results", maxsize=20, drop_oldest=True,
        )

        logger.info("Queue infrastructure created (4 bounded queues)")

    def _load_audio_model(self) -> None:
        """Load the AASIST audio model with graceful degradation."""
        start = time.perf_counter()
        try:
            from app.audio.inference.voice_detector import VoiceDetector

            self._voice_detector = VoiceDetector()
            load_time = (time.perf_counter() - start) * 1000.0

            self._audio_model_status = ModelStatus(
                name="AASIST",
                loaded=True,
                device=str(self._config.DEVICE),
                version=self._config.model.model_version,
                load_time_ms=round(load_time, 2),
            )
            logger.info("Audio model loaded in %.1fms", load_time)

        except Exception as exc:
            load_time = (time.perf_counter() - start) * 1000.0
            self._audio_model_status = ModelStatus(
                name="AASIST",
                loaded=False,
                error=str(exc),
                load_time_ms=round(load_time, 2),
            )
            logger.warning("Audio model load failed (degraded mode): %s", exc)

    def _load_video_model(self) -> None:
        """Load the video deepfake detector with graceful degradation."""
        start = time.perf_counter()
        try:
            from app.video.inference.video_detector import VideoDeepfakeDetector

            self._video_detector = VideoDeepfakeDetector()
            load_time = (time.perf_counter() - start) * 1000.0

            self._video_model_status = ModelStatus(
                name="VideoDeepfakeDetector",
                loaded=True,
                device="cpu",  # heuristic detector is CPU-based
                version=self._config.model.model_version,
                load_time_ms=round(load_time, 2),
            )
            logger.info("Video detector loaded in %.1fms", load_time)

        except Exception as exc:
            load_time = (time.perf_counter() - start) * 1000.0
            self._video_model_status = ModelStatus(
                name="VideoDeepfakeDetector",
                loaded=False,
                error=str(exc),
                load_time_ms=round(load_time, 2),
            )
            logger.warning("Video detector load failed (degraded mode): %s", exc)

    def _create_workers(self) -> None:
        """Create worker instances (but do not start them)."""
        if self._voice_detector is not None:
            self._audio_worker = AudioWorker(
                voice_detector=self._voice_detector,
                input_queue=self._audio_input_queue,
                result_queue=self._audio_result_queue,
                supervisor=self._supervisor,
                health_monitor=self._health_monitor,
                stop_event=self._stop_event,
            )
            logger.info("AudioWorker created")

        if self._video_detector is not None:
            self._video_worker = VideoWorker(
                video_detector=self._video_detector,
                input_queue=self._video_input_queue,
                result_queue=self._video_result_queue,
                supervisor=self._supervisor,
                health_monitor=self._health_monitor,
                stop_event=self._stop_event,
            )
            logger.info("VideoWorker created")

    # ══════════════════════════════════════════
    # START / STOP
    # ══════════════════════════════════════════

    def start(self) -> bool:
        """Start all workers and the result dispatcher.

        Returns ``True`` on success.  Requires state READY.
        """
        try:
            self._set_state(RuntimeState.RUNNING)
            self._stop_event.clear()
            self._start_time = time.monotonic()

            # Start thread supervisor
            if self._supervisor is not None:
                # Register workers with supervisor for auto-restart
                if self._audio_worker is not None:
                    self._supervisor.register_thread(
                        AudioWorker.WORKER_NAME,
                        target=self._audio_worker.get_loop_target(),
                        max_restarts=5,
                    )
                if self._video_worker is not None:
                    self._supervisor.register_thread(
                        VideoWorker.WORKER_NAME,
                        target=self._video_worker.get_loop_target(),
                        max_restarts=5,
                    )
                self._supervisor.start()

            # Start workers
            if self._audio_worker is not None:
                self._audio_worker.start()
            if self._video_worker is not None:
                self._video_worker.start()

            # Start result dispatcher
            if self._result_dispatcher is not None:
                self._result_dispatcher.start()

            event_bus.publish(ServiceStateEvent(service="RuntimeController", running=True))
            logger.info("Runtime started — all workers active")
            return True

        except Exception as exc:
            self._error_message = f"Start failed: {exc}"
            logger.error(self._error_message, exc_info=True)
            try:
                self._set_state(RuntimeState.ERROR)
            except RuntimeError:
                self._state = RuntimeState.ERROR
            return False

    def stop(self) -> bool:
        """Stop all workers, dispatcher, and supervisor gracefully.

        Returns ``True`` on successful shutdown.
        """
        try:
            self._set_state(RuntimeState.STOPPING)
            self._stop_event.set()

            # Stop workers
            if self._audio_worker is not None:
                self._audio_worker.stop(timeout=3.0)
            if self._video_worker is not None:
                self._video_worker.stop(timeout=3.0)

            # Stop result dispatcher
            if self._result_dispatcher is not None:
                self._result_dispatcher.stop(timeout=3.0)

            # Stop supervisor
            if self._supervisor is not None:
                self._supervisor.stop()

            # Clear queues
            if self._queue_manager is not None:
                self._queue_manager.clear_all()

            event_bus.publish(ServiceStateEvent(service="RuntimeController", running=False))

            self._set_state(RuntimeState.STOPPED)
            logger.info("Runtime stopped cleanly")
            return True

        except Exception as exc:
            self._error_message = f"Stop failed: {exc}"
            logger.error(self._error_message, exc_info=True)
            try:
                self._set_state(RuntimeState.ERROR)
            except RuntimeError:
                self._state = RuntimeState.ERROR
            return False

    # ══════════════════════════════════════════
    # DATA INPUT API (for capture layers)
    # ══════════════════════════════════════════

    def push_audio(self, audio_buffer) -> bool:
        """Push an audio buffer for inference.

        Parameters
        ----------
        audio_buffer : np.ndarray
            Raw audio waveform.

        Returns
        -------
        bool
            True if accepted, False if rejected (not running / no worker).
        """
        if self.state != RuntimeState.RUNNING:
            return False
        if self._audio_input_queue is None or self._audio_worker is None:
            return False
        return self._audio_input_queue.put(audio_buffer)

    def push_video_frames(self, frames) -> bool:
        """Push a video frame batch for inference.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of BGR frames.

        Returns
        -------
        bool
            True if accepted, False if rejected (not running / no worker).
        """
        if self.state != RuntimeState.RUNNING:
            return False
        if self._video_input_queue is None or self._video_worker is None:
            return False
        return self._video_input_queue.put(frames)

    # ══════════════════════════════════════════
    # STATUS API
    # ══════════════════════════════════════════

    def get_status(self) -> RuntimeStatus:
        """Return a complete structured snapshot of the runtime state.

        This is the primary API for the future GUI to query runtime health.
        """
        # Uptime
        uptime = 0.0
        if self._start_time is not None and self.state == RuntimeState.RUNNING:
            uptime = time.monotonic() - self._start_time

        # Worker statuses
        workers: List[WorkerStatus] = []
        now = time.monotonic()

        if self._audio_worker is not None:
            sup_status = self._get_supervisor_status(AudioWorker.WORKER_NAME)
            heartbeat_age = now - sup_status.get("last_heartbeat", now) if sup_status else 0.0
            workers.append(WorkerStatus(
                name=AudioWorker.WORKER_NAME,
                alive=self._audio_worker.is_running,
                heartbeat_age_seconds=round(heartbeat_age, 2),
                restart_count=sup_status.get("restart_count", 0) if sup_status else 0,
                total_inferences=self._audio_worker.total_inferences,
                last_error=self._audio_worker.last_error,
            ))

        if self._video_worker is not None:
            sup_status = self._get_supervisor_status(VideoWorker.WORKER_NAME)
            heartbeat_age = now - sup_status.get("last_heartbeat", now) if sup_status else 0.0
            workers.append(WorkerStatus(
                name=VideoWorker.WORKER_NAME,
                alive=self._video_worker.is_running,
                heartbeat_age_seconds=round(heartbeat_age, 2),
                restart_count=sup_status.get("restart_count", 0) if sup_status else 0,
                total_inferences=self._video_worker.total_inferences,
                last_error=self._video_worker.last_error,
            ))

        # Model statuses
        models = [self._audio_model_status, self._video_model_status]

        # Stream statuses
        streams: List[StreamStatus] = []
        if self._queue_manager is not None:
            all_stats = self._queue_manager.get_all_stats()
            for name in ("audio_input", "video_input"):
                qs = all_stats.get(name)
                if qs is not None:
                    streams.append(StreamStatus(
                        name=name,
                        active=self.state == RuntimeState.RUNNING,
                        queue_size=qs.size,
                        queue_capacity=qs.maxsize,
                        total_items_processed=qs.total_dequeued,
                        drop_count=qs.total_dropped,
                    ))

        # Health summary
        health = HealthSummary()
        if self._health_monitor is not None:
            try:
                snap = self._health_monitor.collect_snapshot()
                health = HealthSummary(
                    cpu_percent=snap.cpu_percent,
                    ram_percent=snap.ram_percent,
                    ram_used_mb=snap.ram_used_mb,
                    gpu_allocated_mb=snap.gpu_allocated_mb,
                    gpu_reserved_mb=snap.gpu_reserved_mb,
                    fps=snap.fps,
                    avg_inference_latency_ms=snap.avg_inference_latency_ms,
                )
            except Exception as exc:
                logger.debug("Health snapshot failed: %s", exc)

        # Latest fused prediction
        latest_fused = None
        if self._result_dispatcher is not None:
            latest_fused = self._result_dispatcher.latest_fused

        return RuntimeStatus(
            state=self.state,
            uptime_seconds=round(uptime, 2),
            workers=workers,
            models=models,
            streams=streams,
            health=health,
            last_fused_prediction=latest_fused,
            error_message=self._error_message,
        )

    def _get_supervisor_status(self, worker_name: str) -> Optional[Dict[str, Any]]:
        """Get supervisor status for a named worker thread."""
        if self._supervisor is None:
            return None
        all_status = self._supervisor.get_status()
        return all_status.get(worker_name)

    # ══════════════════════════════════════════
    # CONVENIENCE
    # ══════════════════════════════════════════

    @property
    def has_audio(self) -> bool:
        """Whether an audio model is loaded and worker is available."""
        return self._audio_model_status.loaded and self._audio_worker is not None

    @property
    def has_video(self) -> bool:
        """Whether a video detector is loaded and worker is available."""
        return self._video_model_status.loaded and self._video_worker is not None

    def __repr__(self) -> str:
        return (
            f"RuntimeController(state={self.state.value}, "
            f"audio={self.has_audio}, video={self.has_video})"
        )
