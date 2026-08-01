"""
Detection Service — orchestrates the real-time multimodal deepfake pipeline.

State Machine
─────────────
    IDLE  →  (call detected)  →  WARMUP_15S  →  (15s elapsed)  →  CONTINUOUS_5S
                                                                    ↑ (every 5s) ↓
                                                                    └─ re-eval ──┘
         ←  (call ended / stop pressed)  ←  IDLE

Threading Model
───────────────
• Main thread:     PyQt6 GUI (signal/slot updates)
• Worker threads:  AudioCapture, VideoCaptureWorker, ProcessMonitorWorker
• Inference pool:  ThreadPoolExecutor for off-thread model forward passes

All results are delivered via PyQt6 signals for thread-safe GUI updates.
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from app.fusion.inference.fusion_engine import MultimodalFusion
from app.video.inference.video_detector import VideoDeepfakeDetector
from app.audio.inference.voice_detector import VoiceDetector
from app.config.settings import settings
from app.monitoring.audio_capture import AudioCapture
from app.monitoring.process_monitor import ProcessMonitor
from app.monitoring.screen_capture import ScreenCapture
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DetectionState(enum.Enum):
    """Detection pipeline state machine states."""
    IDLE = "IDLE"
    WARMUP_15S = "WARMUP_15S"
    CONTINUOUS_5S = "CONTINUOUS_5S"


class DetectionService(QObject):
    """
    Orchestrates process monitoring, audio/video capture, and AI inference.

    Emits PyQt6 signals for thread-safe GUI updates:
    - ``call_detected(str)`` — when a video call app is found/lost
    - ``buffer_progress(int)`` — warmup buffer fill percentage (0–100)
    - ``analysis_result(dict)`` — full fusion result dict
    - ``status_message(str)`` — human-readable log messages

    Parameters
    ----------
    parent : QObject, optional
        Parent Qt object for signal ownership.
    """

    # ── PyQt6 Signals ─────────────────────────
    call_detected = pyqtSignal(str)        # "WhatsApp" or "" when lost
    buffer_progress = pyqtSignal(int)      # 0–100
    analysis_result = pyqtSignal(dict)     # fusion result dict
    status_message = pyqtSignal(str)       # log messages

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        # ── State ─────────────────────────────
        self._state = DetectionState.IDLE
        self._active_app: str = ""
        self._is_running = False

        # ── Sub-systems ───────────────────────
        self._process_monitor = ProcessMonitor()
        self._audio_capture = AudioCapture(
            sample_rate=settings.audio.sample_rate,
            buffer_duration=float(settings.INITIAL_WARMUP_SEC),
        )
        self._screen_capture = ScreenCapture(
            target_fps=settings.VIDEO_TARGET_FPS,
            buffer_duration=float(settings.INITIAL_WARMUP_SEC),
        )

        # ── AI Components ─────────────────────
        self._voice_detector = VoiceDetector()
        self._video_detector = VideoDeepfakeDetector()
        self._fusion = MultimodalFusion()

        # ── Thread Pool for inference ─────────
        self._inference_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="InferenceWorker"
        )

        # ── Timers ────────────────────────────
        self._process_timer: Optional[QTimer] = None    # Scans every 3s
        self._warmup_timer: Optional[QTimer] = None     # Warmup progress check
        self._eval_timer: Optional[QTimer] = None       # 5s eval cycle

        # ── Worker threads ────────────────────
        self._video_capture_thread: Optional[threading.Thread] = None
        self._video_stop_event = threading.Event()

        # ── Warmup tracking ───────────────────
        self._warmup_start_time: float = 0.0

    @property
    def state(self) -> DetectionState:
        """Current detection pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Whether detection is active."""
        return self._is_running

    # ══════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════

    def start_monitoring(self) -> None:
        """Start process monitoring and detection pipeline."""
        if self._is_running:
            self.status_message.emit("Detection already running")
            return

        self._is_running = True
        self._state = DetectionState.IDLE
        self.status_message.emit("Monitoring started — scanning for video calls...")

        # Start process monitor timer (scan every 3 seconds)
        self._process_timer = QTimer(self)
        self._process_timer.timeout.connect(self._scan_processes)
        self._process_timer.start(3000)

        # Trigger immediate scan
        self._scan_processes()

    def stop_monitoring(self) -> None:
        """Stop all monitoring, capture, and inference."""
        if not self._is_running:
            return

        self._is_running = False
        self._stop_all_timers()
        self._stop_capture()
        self._state = DetectionState.IDLE
        self._active_app = ""
        self.call_detected.emit("")
        self.buffer_progress.emit(0)
        self.status_message.emit("Monitoring stopped")
        logger.info("DetectionService stopped")

    # ══════════════════════════════════════════
    # PROCESS MONITORING
    # ══════════════════════════════════════════

    def _scan_processes(self) -> None:
        """Scan for active video call processes."""
        try:
            detected = self._process_monitor.scan_processes()
            if detected:
                # Pick the first detected app
                app_name = detected[0].display_name
                proc_name = detected[0].name

                if app_name != self._active_app:
                    self._active_app = app_name
                    self.call_detected.emit(app_name)
                    self.status_message.emit(f"Call detected: {app_name}")
                    logger.info("Video call detected: %s (%s)", app_name, proc_name)

                    # Try to get window geometry for targeted capture
                    geometry = self._process_monitor.get_window_geometry(proc_name)
                    if geometry:
                        self._screen_capture.set_region(geometry)
                        self.status_message.emit(
                            f"Window captured: {geometry[2]}×{geometry[3]} at ({geometry[0]}, {geometry[1]})"
                        )

                # Start capture if not already running
                if self._state == DetectionState.IDLE:
                    self._start_warmup()

            else:
                if self._active_app:
                    self._active_app = ""
                    self.call_detected.emit("")
                    self.status_message.emit("Call ended — returning to idle")
                    self._stop_capture()
                    self._state = DetectionState.IDLE

        except Exception as exc:
            logger.error("Process scan error: %s", exc)

    # ══════════════════════════════════════════
    # PHASE 1: WARMUP (15 seconds)
    # ══════════════════════════════════════════

    def _start_warmup(self) -> None:
        """Begin 15-second warmup: start audio + video capture."""
        self._state = DetectionState.WARMUP_15S
        self._warmup_start_time = time.time()
        self.status_message.emit("Phase 1: Recording 15-second warmup buffer...")
        self.buffer_progress.emit(0)

        # Clear old buffers
        self._audio_capture.clear_buffer()
        self._screen_capture.clear_buffer()

        # Start audio capture (WASAPI loopback)
        self._audio_capture.start()

        # Start video capture thread (mss at 5 FPS)
        self._video_stop_event.clear()
        self._video_capture_thread = threading.Thread(
            target=self._video_capture_loop,
            name="VideoCaptureWorker",
            daemon=True,
        )
        self._video_capture_thread.start()

        # Start warmup progress timer (update every 250ms for smooth progress)
        self._warmup_timer = QTimer(self)
        self._warmup_timer.timeout.connect(self._check_warmup_progress)
        self._warmup_timer.start(250)

    def _check_warmup_progress(self) -> None:
        """Check warmup buffer fill and transition to continuous mode."""
        elapsed = time.time() - self._warmup_start_time
        progress = min(int((elapsed / settings.INITIAL_WARMUP_SEC) * 100), 100)
        self.buffer_progress.emit(progress)

        if elapsed >= settings.INITIAL_WARMUP_SEC:
            # Warmup complete — run initial evaluation
            if self._warmup_timer:
                self._warmup_timer.stop()
            self.buffer_progress.emit(100)
            self.status_message.emit("Warmup complete — running initial analysis...")
            self._run_evaluation()
            self._enter_continuous_mode()

    # ══════════════════════════════════════════
    # PHASE 2: CONTINUOUS (5-second rolling)
    # ══════════════════════════════════════════

    def _enter_continuous_mode(self) -> None:
        """Transition to continuous 5-second rolling evaluation."""
        self._state = DetectionState.CONTINUOUS_5S
        self.status_message.emit("Phase 2: Continuous 5-second rolling evaluation active")

        # Start evaluation timer (every 5 seconds)
        self._eval_timer = QTimer(self)
        self._eval_timer.timeout.connect(self._run_evaluation)
        self._eval_timer.start(settings.INCREMENTAL_CHUNK_SEC * 1000)

    def _run_evaluation(self) -> None:
        """Submit inference task to the thread pool."""
        if not self._is_running:
            return

        # Grab current 15-second buffer snapshots
        audio_buffer = self._audio_capture.get_15s_buffer()
        video_buffer = self._screen_capture.get_15s_buffer()

        # Submit to inference thread pool (non-blocking)
        self._inference_pool.submit(
            self._evaluate_buffers, audio_buffer, video_buffer
        )

    def _evaluate_buffers(
        self, audio_buffer, video_buffer
    ) -> None:
        """
        Run audio + video inference and emit fused result.

        Runs in a ThreadPoolExecutor worker thread.
        """
        try:
            start = time.perf_counter()

            # Audio analysis
            audio_score = 0.5
            if len(audio_buffer) > 0:
                audio_score = self._voice_detector.predict_from_buffer(
                    audio_buffer, sr=settings.audio.sample_rate
                )

            # Video analysis
            video_score = 0.5
            if video_buffer and len(video_buffer) > 0:
                video_score = self._video_detector.predict_from_frames(video_buffer)

            # Fusion
            result = self._fusion.evaluate(audio_score, video_score)

            elapsed = (time.perf_counter() - start) * 1000
            result["latency_ms"] = round(elapsed, 1)
            result["audio_buffer_samples"] = len(audio_buffer)
            result["video_buffer_frames"] = len(video_buffer) if video_buffer else 0
            result["state"] = self._state.value

            # Emit signal (thread-safe via Qt's signal mechanism)
            self.analysis_result.emit(result)

            logger.info(
                "Evaluation: %s (combined=%.4f, audio=%.4f, video=%.4f, %.1fms)",
                result["prediction"],
                result["combined_score"],
                audio_score, video_score, elapsed,
            )

        except Exception as exc:
            logger.error("Evaluation error: %s", exc)
            self.status_message.emit(f"Analysis error: {exc}")

    # ══════════════════════════════════════════
    # VIDEO CAPTURE WORKER
    # ══════════════════════════════════════════

    def _video_capture_loop(self) -> None:
        """Background thread: captures screen frames at target FPS."""
        interval = 1.0 / self._screen_capture.target_fps
        logger.info("Video capture thread started (%.1f FPS)", self._screen_capture.target_fps)

        while not self._video_stop_event.is_set() and self._is_running:
            try:
                self._screen_capture.capture_to_buffer()
            except Exception as exc:
                logger.debug("Frame capture error: %s", exc)

            self._video_stop_event.wait(interval)

        logger.info("Video capture thread stopped")

    # ══════════════════════════════════════════
    # CLEANUP
    # ══════════════════════════════════════════

    def _stop_all_timers(self) -> None:
        """Stop all QTimers."""
        for timer in [self._process_timer, self._warmup_timer, self._eval_timer]:
            if timer is not None:
                timer.stop()
        self._process_timer = None
        self._warmup_timer = None
        self._eval_timer = None

    def _stop_capture(self) -> None:
        """Stop audio and video capture."""
        # Stop audio
        self._audio_capture.stop()

        # Stop video capture thread
        self._video_stop_event.set()
        if self._video_capture_thread is not None:
            self._video_capture_thread.join(timeout=2.0)
            self._video_capture_thread = None

        # Close screen capture resources
        self._screen_capture.close()

    def cleanup(self) -> None:
        """Full cleanup — call before application exit."""
        self.stop_monitoring()
        self._inference_pool.shutdown(wait=False)
        logger.info("DetectionService cleaned up")
