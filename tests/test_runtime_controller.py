"""
Production test suite for the Runtime Orchestration Layer.

Tests cover:
1. RuntimeStatus dataclasses and serialisation
2. RuntimeController state machine transitions
3. Audio model loading with graceful degradation
4. Video model loading with graceful degradation
5. AudioWorker lifecycle and inference flow
6. VideoWorker lifecycle and inference flow
7. ResultDispatcher fusion synchronisation
8. ResultDispatcher single-modality fallback
9. Full RuntimeController initialize → start → stop lifecycle
10. Push API (push_audio / push_video_frames)
11. Structured get_status() API
12. Missing stream handling (audio-only / video-only)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.interfaces import DetectionLabel, Modality, PredictionResult
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


# ══════════════════════════════════════════════
# 1. RuntimeStatus Dataclasses
# ══════════════════════════════════════════════


class TestRuntimeStatus:
    """Test the structured status types and serialisation."""

    def test_runtime_state_enum_values(self):
        """All expected states exist."""
        assert RuntimeState.UNINITIALIZED.value == "UNINITIALIZED"
        assert RuntimeState.INITIALIZING.value == "INITIALIZING"
        assert RuntimeState.READY.value == "READY"
        assert RuntimeState.RUNNING.value == "RUNNING"
        assert RuntimeState.STOPPING.value == "STOPPING"
        assert RuntimeState.STOPPED.value == "STOPPED"
        assert RuntimeState.ERROR.value == "ERROR"

    def test_worker_status_healthy(self):
        """WorkerStatus.healthy reflects alive + heartbeat age."""
        w = WorkerStatus(name="test", alive=True, heartbeat_age_seconds=2.0)
        assert w.healthy is True

        w2 = WorkerStatus(name="test", alive=True, heartbeat_age_seconds=15.0)
        assert w2.healthy is False

        w3 = WorkerStatus(name="test", alive=False, heartbeat_age_seconds=1.0)
        assert w3.healthy is False

    def test_model_status_defaults(self):
        """ModelStatus defaults are sensible."""
        m = ModelStatus(name="AASIST")
        assert m.loaded is False
        assert m.device == "unknown"

    def test_runtime_status_to_dict(self):
        """RuntimeStatus.to_dict() produces a serialisable dictionary."""
        status = RuntimeStatus(
            state=RuntimeState.RUNNING,
            uptime_seconds=42.5,
            workers=[WorkerStatus(name="AudioWorker", alive=True)],
            models=[ModelStatus(name="AASIST", loaded=True, device="cuda:0")],
            streams=[StreamStatus(name="audio_input", active=True, queue_size=3, queue_capacity=10)],
            health=HealthSummary(cpu_percent=25.0, ram_percent=60.0),
        )
        d = status.to_dict()

        assert d["state"] == "RUNNING"
        assert d["uptime_seconds"] == 42.5
        assert len(d["workers"]) == 1
        assert d["workers"][0]["name"] == "AudioWorker"
        assert d["workers"][0]["alive"] is True
        assert len(d["models"]) == 1
        assert d["models"][0]["loaded"] is True
        assert d["health"]["cpu_percent"] == 25.0

    def test_stream_status_fields(self):
        """StreamStatus holds queue metrics."""
        s = StreamStatus(
            name="video_input", active=True,
            queue_size=5, queue_capacity=20,
            total_items_processed=100, drop_count=2,
        )
        assert s.active is True
        assert s.drop_count == 2


# ══════════════════════════════════════════════
# 2. Mock Detectors for Worker Tests
# ══════════════════════════════════════════════


class MockVoiceDetector:
    """Mock voice detector that returns a fixed fake probability."""

    def __init__(self, fake_prob: float = 0.85) -> None:
        self.fake_prob = fake_prob
        self.call_count = 0

    def predict_buffer(self, audio_buffer: np.ndarray) -> float:
        self.call_count += 1
        return self.fake_prob


class MockVideoDetector:
    """Mock video detector that returns a fixed fake probability."""

    def __init__(self, fake_prob: float = 0.3) -> None:
        self.fake_prob = fake_prob
        self.call_count = 0

    def predict_from_frames(self, frames: list) -> float:
        self.call_count += 1
        return self.fake_prob


class FailingVoiceDetector:
    """Mock detector that raises an exception."""

    def predict_buffer(self, audio_buffer: np.ndarray) -> float:
        raise RuntimeError("Simulated audio inference failure")


# ══════════════════════════════════════════════
# 3. AudioWorker Tests
# ══════════════════════════════════════════════


class TestAudioWorker:
    """Test AudioWorker lifecycle and inference flow."""

    def _make_worker(self, detector=None):
        """Create a worker with mock detector and fresh queues."""
        det = detector or MockVoiceDetector(fake_prob=0.85)
        input_q = BoundedQueue(name="audio_in", maxsize=10)
        result_q = BoundedQueue(name="audio_out", maxsize=10)
        worker = AudioWorker(
            voice_detector=det,
            input_queue=input_q,
            result_queue=result_q,
        )
        return worker, input_q, result_q, det

    def test_start_stop_lifecycle(self):
        """Worker starts and stops cleanly."""
        worker, _, _, _ = self._make_worker()

        assert not worker.is_running
        worker.start()
        assert worker.is_running
        worker.stop(timeout=2.0)
        assert not worker.is_running

    def test_inference_produces_result(self):
        """Pushing audio into the input queue produces a PredictionResult."""
        worker, input_q, result_q, det = self._make_worker()
        worker.start()

        # Push a mock audio buffer
        audio = np.random.randn(16000).astype(np.float32)
        input_q.put(audio)

        # Wait for result
        result = result_q.get(block=True, timeout=3.0)
        worker.stop(timeout=2.0)

        assert result is not None
        assert isinstance(result, PredictionResult)
        assert result.modality == Modality.AUDIO
        assert result.confidence == 0.85
        assert result.label == DetectionLabel.FAKE
        assert worker.total_inferences == 1
        assert det.call_count == 1

    def test_multiple_inferences(self):
        """Worker processes multiple audio buffers sequentially."""
        worker, input_q, result_q, det = self._make_worker()
        worker.start()

        for _ in range(5):
            input_q.put(np.random.randn(16000).astype(np.float32))

        time.sleep(1.0)
        worker.stop(timeout=2.0)

        assert det.call_count == 5
        assert worker.total_inferences == 5

    def test_error_recovery(self):
        """Worker survives a failing detector without crashing the thread."""
        worker, input_q, result_q, _ = self._make_worker(
            detector=FailingVoiceDetector()
        )
        worker.start()

        input_q.put(np.random.randn(16000).astype(np.float32))
        time.sleep(0.5)

        # Worker should still be alive after error
        assert worker.is_running
        assert worker.last_error is not None
        assert "Simulated" in worker.last_error
        assert worker.total_inferences == 0

        worker.stop(timeout=2.0)

    def test_health_monitor_integration(self):
        """Worker records latency in HealthMonitor."""
        qm = QueueManager()
        hm = HealthMonitor(queue_manager=qm)
        det = MockVoiceDetector(fake_prob=0.5)
        input_q = BoundedQueue(name="audio_in", maxsize=10)
        result_q = BoundedQueue(name="audio_out", maxsize=10)

        worker = AudioWorker(
            voice_detector=det,
            input_queue=input_q,
            result_queue=result_q,
            health_monitor=hm,
        )
        worker.start()

        input_q.put(np.random.randn(16000).astype(np.float32))
        time.sleep(0.5)
        worker.stop(timeout=2.0)

        assert hm.get_average_latency_ms() > 0.0


# ══════════════════════════════════════════════
# 4. VideoWorker Tests
# ══════════════════════════════════════════════


class TestVideoWorker:
    """Test VideoWorker lifecycle and inference flow."""

    def _make_worker(self, detector=None):
        det = detector or MockVideoDetector(fake_prob=0.3)
        input_q = BoundedQueue(name="video_in", maxsize=10)
        result_q = BoundedQueue(name="video_out", maxsize=10)
        worker = VideoWorker(
            video_detector=det,
            input_queue=input_q,
            result_queue=result_q,
        )
        return worker, input_q, result_q, det

    def test_start_stop_lifecycle(self):
        worker, _, _, _ = self._make_worker()
        assert not worker.is_running
        worker.start()
        assert worker.is_running
        worker.stop(timeout=2.0)
        assert not worker.is_running

    def test_inference_produces_result(self):
        worker, input_q, result_q, det = self._make_worker()
        worker.start()

        # Push mock video frames
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        input_q.put(frames)

        result = result_q.get(block=True, timeout=3.0)
        worker.stop(timeout=2.0)

        assert result is not None
        assert isinstance(result, PredictionResult)
        assert result.modality == Modality.VIDEO
        assert result.confidence == 0.3
        assert result.label == DetectionLabel.REAL
        assert worker.total_inferences == 1

    def test_uncertain_classification(self):
        """Score between thresholds produces UNCERTAIN label."""
        worker, input_q, result_q, _ = self._make_worker(
            detector=MockVideoDetector(fake_prob=0.5)
        )
        worker.start()

        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        input_q.put(frames)

        result = result_q.get(block=True, timeout=3.0)
        worker.stop(timeout=2.0)

        assert result.label == DetectionLabel.UNCERTAIN


# ══════════════════════════════════════════════
# 5. ResultDispatcher Tests
# ══════════════════════════════════════════════


class TestResultDispatcher:
    """Test prediction synchronisation and fusion dispatch."""

    def _make_dispatcher(self, sync_window: float = 5.0):
        audio_q = BoundedQueue(name="audio_results", maxsize=20)
        video_q = BoundedQueue(name="video_results", maxsize=20)
        fusion = MultimodalFusion()
        stop = threading.Event()
        dispatcher = ResultDispatcher(
            audio_result_queue=audio_q,
            video_result_queue=video_q,
            fusion_engine=fusion,
            sync_window_sec=sync_window,
            stop_event=stop,
        )
        return dispatcher, audio_q, video_q, stop

    def test_start_stop(self):
        dispatcher, _, _, _ = self._make_dispatcher()
        dispatcher.start()
        assert dispatcher.is_running
        dispatcher.stop(timeout=2.0)
        assert not dispatcher.is_running

    def test_fusion_when_both_modalities(self):
        """Dispatcher fuses when both audio and video results arrive."""
        dispatcher, audio_q, video_q, _ = self._make_dispatcher()
        dispatcher.start()

        audio_pred = PredictionResult(
            label=DetectionLabel.FAKE, confidence=0.85,
            modality=Modality.AUDIO, latency_ms=10.0,
        )
        video_pred = PredictionResult(
            label=DetectionLabel.REAL, confidence=0.3,
            modality=Modality.VIDEO, latency_ms=15.0,
        )

        audio_q.put(audio_pred)
        video_q.put(video_pred)

        time.sleep(0.5)
        dispatcher.stop(timeout=2.0)

        assert dispatcher.total_fusions >= 1
        latest = dispatcher.latest_fused
        assert latest is not None
        assert "combined_score" in latest

    def test_single_modality_fallback(self):
        """Dispatcher publishes audio-only when video is absent."""
        dispatcher, audio_q, video_q, _ = self._make_dispatcher()
        dispatcher.start()

        audio_pred = PredictionResult(
            label=DetectionLabel.FAKE, confidence=0.9,
            modality=Modality.AUDIO, latency_ms=10.0,
        )
        audio_q.put(audio_pred)

        time.sleep(0.5)
        dispatcher.stop(timeout=2.0)

        latest = dispatcher.latest_fused
        assert latest is not None
        assert latest.get("modality") == "audio" or "combined_score" in latest

    def test_stale_prediction_not_fused(self):
        """Predictions older than sync window are not fused."""
        dispatcher, audio_q, video_q, _ = self._make_dispatcher(sync_window=0.3)
        dispatcher.start()

        # Push audio result
        audio_pred = PredictionResult(
            label=DetectionLabel.FAKE, confidence=0.8,
            modality=Modality.AUDIO, latency_ms=5.0,
        )
        audio_q.put(audio_pred)

        # Wait for it to become stale
        time.sleep(0.5)

        # Push video result (audio is now stale)
        video_pred = PredictionResult(
            label=DetectionLabel.REAL, confidence=0.2,
            modality=Modality.VIDEO, latency_ms=8.0,
        )
        video_q.put(video_pred)

        time.sleep(0.3)
        dispatcher.stop(timeout=2.0)

        # Should be video-only, not fused (since audio is stale)
        assert dispatcher.total_fusions == 0


# ══════════════════════════════════════════════
# 6. RuntimeController Tests
# ══════════════════════════════════════════════


class TestRuntimeController:
    """Test the central orchestrator lifecycle and APIs."""

    def test_initial_state(self):
        """Controller starts in UNINITIALIZED state."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        assert rc.state == RuntimeState.UNINITIALIZED

    def test_initialize_reaches_ready(self):
        """Controller transitions to READY after successful init."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        result = rc.initialize()
        assert result is True
        assert rc.state == RuntimeState.READY

    def test_full_lifecycle(self):
        """Full init → start → stop lifecycle with correct state transitions."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()

        assert rc.initialize() is True
        assert rc.state == RuntimeState.READY

        assert rc.start() is True
        assert rc.state == RuntimeState.RUNNING

        time.sleep(0.3)

        assert rc.stop() is True
        assert rc.state == RuntimeState.STOPPED

    def test_get_status_uninitialized(self):
        """get_status() works even before initialization."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        status = rc.get_status()
        assert status.state == RuntimeState.UNINITIALIZED
        assert status.uptime_seconds == 0.0

    def test_get_status_running(self):
        """get_status() returns populated status when running."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()
        rc.start()

        time.sleep(0.3)
        status = rc.get_status()

        assert status.state == RuntimeState.RUNNING
        assert status.uptime_seconds > 0
        assert len(status.models) == 2  # audio + video
        assert len(status.workers) >= 1  # at least one worker

        # Verify to_dict works
        d = status.to_dict()
        assert d["state"] == "RUNNING"

        rc.stop()

    def test_push_audio_when_running(self):
        """push_audio() accepts data when running."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()
        rc.start()

        audio = np.random.randn(16000).astype(np.float32)
        accepted = rc.push_audio(audio)

        rc.stop()

        if rc.has_audio:
            assert accepted is True
        else:
            # If audio model failed to load, push returns False
            assert accepted is False

    def test_push_audio_when_not_running(self):
        """push_audio() rejects data when not running."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()

        audio = np.random.randn(16000).astype(np.float32)
        assert rc.push_audio(audio) is False

    def test_push_video_when_running(self):
        """push_video_frames() accepts data when running."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()
        rc.start()

        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        accepted = rc.push_video_frames(frames)

        rc.stop()

        if rc.has_video:
            assert accepted is True
        else:
            assert accepted is False

    def test_has_audio_and_video(self):
        """has_audio and has_video reflect model loading status."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()

        # At minimum, models should attempt to load
        assert isinstance(rc.has_audio, bool)
        assert isinstance(rc.has_video, bool)

    def test_invalid_state_transition(self):
        """Attempting invalid transitions raises RuntimeError."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()

        # Can't go from UNINITIALIZED → RUNNING directly
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            rc._set_state(RuntimeState.RUNNING)

    def test_repr(self):
        """Controller has a useful repr."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        r = repr(rc)
        assert "RuntimeController" in r
        assert "UNINITIALIZED" in r

    def test_double_start(self):
        """Starting when already RUNNING is caught and returns False."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()
        rc.start()

        # Second start should return False (RUNNING → RUNNING is invalid)
        # The controller catches the error internally and transitions to ERROR
        result = rc.start()
        assert result is False
        assert rc.state == RuntimeState.ERROR

    def test_reinitialize_after_stop(self):
        """Controller can be re-initialized after stopping."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()
        rc.start()
        rc.stop()

        # STOPPED → INITIALIZING is valid
        assert rc.initialize() is True
        assert rc.state == RuntimeState.READY

    def test_status_models_info(self):
        """Status includes model details (name, loaded, device, version)."""
        from app.runtime.runtime_controller import RuntimeController
        rc = RuntimeController()
        rc.initialize()

        status = rc.get_status()
        model_names = [m.name for m in status.models]
        assert "AASIST" in model_names
        assert "VideoDeepfakeDetector" in model_names

        for m in status.models:
            assert isinstance(m.load_time_ms, float)
            assert m.load_time_ms >= 0


# ══════════════════════════════════════════════
# 7. Integration: End-to-End Data Flow
# ══════════════════════════════════════════════


class TestEndToEndDataFlow:
    """Test data flowing from input queues through workers to the dispatcher."""

    def test_audio_data_flow_to_result(self):
        """Audio buffer → AudioWorker → result queue → PredictionResult."""
        det = MockVoiceDetector(fake_prob=0.9)
        input_q = BoundedQueue(name="audio_in", maxsize=10)
        result_q = BoundedQueue(name="audio_out", maxsize=10)

        worker = AudioWorker(
            voice_detector=det, input_queue=input_q, result_queue=result_q,
        )
        worker.start()

        input_q.put(np.random.randn(16000).astype(np.float32))
        result = result_q.get(block=True, timeout=3.0)

        worker.stop(timeout=2.0)

        assert result is not None
        assert result.confidence == 0.9
        assert result.label == DetectionLabel.FAKE

    def test_video_data_flow_to_result(self):
        """Video frames → VideoWorker → result queue → PredictionResult."""
        det = MockVideoDetector(fake_prob=0.15)
        input_q = BoundedQueue(name="video_in", maxsize=10)
        result_q = BoundedQueue(name="video_out", maxsize=10)

        worker = VideoWorker(
            video_detector=det, input_queue=input_q, result_queue=result_q,
        )
        worker.start()

        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        input_q.put(frames)

        result = result_q.get(block=True, timeout=3.0)
        worker.stop(timeout=2.0)

        assert result is not None
        assert result.confidence == 0.15
        assert result.label == DetectionLabel.REAL

    def test_dual_modality_through_dispatcher(self):
        """Audio + Video workers → dispatcher → fused result."""
        audio_det = MockVoiceDetector(fake_prob=0.8)
        video_det = MockVideoDetector(fake_prob=0.4)

        audio_in = BoundedQueue(name="audio_in", maxsize=10)
        video_in = BoundedQueue(name="video_in", maxsize=10)
        audio_out = BoundedQueue(name="audio_out", maxsize=20)
        video_out = BoundedQueue(name="video_out", maxsize=20)

        stop = threading.Event()

        audio_worker = AudioWorker(
            voice_detector=audio_det, input_queue=audio_in,
            result_queue=audio_out, stop_event=stop,
        )
        video_worker = VideoWorker(
            video_detector=video_det, input_queue=video_in,
            result_queue=video_out, stop_event=stop,
        )

        fusion = MultimodalFusion()
        dispatcher = ResultDispatcher(
            audio_result_queue=audio_out,
            video_result_queue=video_out,
            fusion_engine=fusion,
            sync_window_sec=5.0,
            stop_event=stop,
        )

        # Start everything
        audio_worker.start()
        video_worker.start()
        dispatcher.start()

        # Push data
        audio_in.put(np.random.randn(16000).astype(np.float32))
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        video_in.put(frames)

        # Wait for processing
        time.sleep(1.5)

        # Stop everything
        stop.set()
        audio_worker.stop(timeout=2.0)
        video_worker.stop(timeout=2.0)
        dispatcher.stop(timeout=2.0)

        # Verify fusion happened
        assert dispatcher.total_fusions >= 1
        latest = dispatcher.latest_fused
        assert latest is not None
        assert "combined_score" in latest
        # Expected: 0.60 * 0.8 + 0.40 * 0.4 = 0.64
        assert 0.5 < latest["combined_score"] < 0.8
