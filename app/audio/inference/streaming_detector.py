"""
Real-time streaming audio deepfake detector.

This module bridges the MicrophoneService (audio capture) with the
Predictor (model inference) via a background thread.

Architecture
------------
MicrophoneService  ──on_audio callback──►  StreamingDetector
                                                │
                                                ▼
                               FeatureExtractor (mel spectrogram)
                                                │
                                                ▼
                               Predictor (CNN forward pass)
                                                │
                                                ▼
                               EMA Confidence Tracker (window=5)
                                                │
                                                ▼
                               EventBus.publish(DetectionEvent)
                                                │
                                                ▼
                               GUI (QTimer polls EventBus)

The audio callback from MicrophoneService is non-blocking — it queues
the audio segment and returns immediately.  The inference thread picks
up from the queue and runs the model, preventing audio drop-outs.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from app.audio.inference.predictor import Predictor
from app.audio.preprocessing.audio_preprocessor import AudioPreprocessor
from app.audio.features.feature_extractor import FeatureExtractor
from app.config.settings import settings
from app.core.interfaces import DetectionLabel, Modality, PredictionResult
from app.services.event_bus import DetectionEvent, ServiceStateEvent, StatusEvent, event_bus
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingDetector:
    """
    Real-time audio deepfake detection from a live microphone stream.

    Accepts audio segments from ``MicrophoneService.on_audio`` callback,
    extracts Mel spectrogram features, runs the CNN, and publishes
    ``DetectionEvent`` on the EventBus for the GUI.

    Parameters
    ----------
    predictor : Predictor
        Loaded audio deepfake predictor.
    ema_alpha : float
        EMA smoothing factor for confidence scores (0 = no smoothing,
        1 = only latest value).
    min_detection_interval : float
        Minimum seconds between published DetectionEvents (rate limiting).
    """

    def __init__(
        self,
        predictor: Predictor,
        ema_alpha: float = 0.4,
        min_detection_interval: float = 0.5,
    ) -> None:
        self.predictor = predictor
        self.ema_alpha = ema_alpha
        self.min_detection_interval = min_detection_interval

        # Preprocessing components (reused across calls for efficiency)
        self._preprocessor = AudioPreprocessor()
        self._extractor = FeatureExtractor()

        # Audio segment queue (MicrophoneService → inference thread)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=5)

        # Inference thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # EMA confidence state
        self._ema_confidence: float = 0.0
        self._last_publish_time: float = 0.0

        # Recent predictions for stats
        self._recent: Deque[DetectionLabel] = deque(maxlen=10)

    # ── Properties ────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Public API ────────────────────────────

    def start(self) -> None:
        """Start the inference background thread."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="StreamingDetector",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        event_bus.publish(ServiceStateEvent(service="StreamingDetector", running=True))
        logger.info("StreamingDetector started")

    def stop(self) -> None:
        """Stop the inference thread gracefully."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        event_bus.publish(ServiceStateEvent(service="StreamingDetector", running=False))
        logger.info("StreamingDetector stopped")

    def on_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Callback for MicrophoneService — enqueue audio for inference.

        This is called from the audio thread and must return quickly.

        Parameters
        ----------
        audio : np.ndarray
            Raw audio waveform (float32, mono).
        sample_rate : int
            Audio sample rate in Hz.
        """
        try:
            self._audio_queue.put_nowait((audio, sample_rate))
        except queue.Full:
            # Drop oldest if queue full (prefer fresh audio)
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            self._audio_queue.put_nowait((audio, sample_rate))

    # ── Inference thread ──────────────────────

    def _inference_loop(self) -> None:
        """Pull audio from queue and run detection in a loop."""
        import torch
        import torch.nn.functional as F

        while not self._stop_event.is_set():
            try:
                audio, sr = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self._run_inference(audio)
                self._update_ema(result)
            except Exception as exc:
                logger.warning("Streaming inference error: %s", exc)

    def _run_inference(self, audio: np.ndarray) -> PredictionResult:
        """
        Extract features and run the model on a raw audio array.

        Returns a PredictionResult without EMA smoothing.
        """
        import io
        import tempfile
        import os
        import soundfile as sf

        start = time.perf_counter()

        # Write audio to a temp file for AudioPreprocessor
        # (future: pass ndarray directly to FeatureExtractor)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            sf.write(tmp_path, audio, settings.audio.sample_rate)
            waveform, _ = self._preprocessor.preprocess(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        feature = self._extractor.extract(waveform)

        import torch
        import torch.nn.functional as F

        self.predictor.model.eval()
        with torch.no_grad():
            inp = feature.unsqueeze(0).to(self.predictor.device)
            output = self.predictor.model(inp)
            probs = F.softmax(output, dim=1)
            spoof_prob = probs[0, 1].item()

        latency_ms = (time.perf_counter() - start) * 1000

        if spoof_prob >= self.predictor.threshold_fake:
            label = DetectionLabel.FAKE
        elif spoof_prob <= self.predictor.threshold_real:
            label = DetectionLabel.REAL
        else:
            label = DetectionLabel.UNCERTAIN

        return PredictionResult(
            label=label,
            confidence=spoof_prob,
            modality=Modality.AUDIO,
            latency_ms=round(latency_ms, 2),
            model_version=settings.model.model_version,
        )

    def _update_ema(self, result: PredictionResult) -> None:
        """Update EMA confidence and publish event if interval exceeded."""
        # EMA smoothing
        self._ema_confidence = (
            self.ema_alpha * result.confidence
            + (1 - self.ema_alpha) * self._ema_confidence
        )

        self._recent.append(result.label)

        # Rate-limit publishing
        now = time.monotonic()
        if now - self._last_publish_time < self.min_detection_interval:
            return
        self._last_publish_time = now

        # Build smoothed result
        threshold_fake = settings.inference.confidence_threshold_fake
        threshold_real = settings.inference.confidence_threshold_real
        conf = self._ema_confidence

        if conf >= threshold_fake:
            label = DetectionLabel.FAKE
        elif conf <= threshold_real:
            label = DetectionLabel.REAL
        else:
            label = DetectionLabel.UNCERTAIN

        smoothed = PredictionResult(
            label=label,
            confidence=conf,
            modality=Modality.AUDIO,
            latency_ms=result.latency_ms,
            model_version=result.model_version,
        )

        event_bus.publish(DetectionEvent(result=smoothed))
        logger.debug(
            "Streaming detection: %s (ema_conf=%.3f, latency=%.1fms)",
            label.value, conf, result.latency_ms,
        )
