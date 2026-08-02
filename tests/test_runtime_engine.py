"""Comprehensive unit tests for Phase 7 Runtime Engine Infrastructure."""

import os
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from app.core.interfaces import DetectionLabel, Modality, PredictionResult
from app.core.queue import BoundedQueue, QueueManager
from app.core.buffer import RingBuffer
from app.monitoring.supervisor import ThreadSupervisor
from app.monitoring.health import HealthMonitor
from app.services.storage import TemporaryStorageService
from app.services.history import HistoryService
from app.services.notification import NotificationEvent, NotificationSeverity, NotificationService
from app.services.orchestration import SessionOrchestrator, SessionState


# 1. QueueManager Tests
def test_queue_manager_and_bounded_queue():
    qm = QueueManager()
    q = qm.create_queue(name="video_frames", maxsize=3, drop_oldest=True)
    assert q.maxsize == 3

    q.put("frame_1")
    q.put("frame_2")
    q.put("frame_3")
    q.put("frame_4")  # Should drop frame_1

    stats = q.get_stats()
    assert stats.total_enqueued == 4
    assert stats.total_dropped == 1
    assert stats.size == 3

    item = q.get(block=False)
    assert item == "frame_2"

    all_stats = qm.get_all_stats()
    assert "video_frames" in all_stats
    qm.clear_all()
    assert q.get_stats().size == 0


# 2. RingBuffer Tests
def test_ring_buffer_audio_and_video():
    buffer = RingBuffer(capacity_samples_or_frames=5, duration_seconds=5.0, sample_rate_or_fps=1.0)
    for i in range(7):
        frame = np.ones((10, 10, 3), dtype=np.uint8) * i
        buffer.append(frame)

    assert len(buffer) == 5
    latest = buffer.get_latest(count=3)
    assert len(latest) == 3
    assert float(latest[-1][0, 0, 0]) == 6.0

    tensor_out = buffer.get_as_tensor()
    assert tensor_out is not None
    assert tensor_out.shape == (5, 10, 10, 3)

    buffer.clear()
    assert len(buffer) == 0


# 3. ThreadSupervisor Tests
def test_thread_supervisor():
    sup = ThreadSupervisor(check_interval=0.1, heartbeat_timeout=0.3)
    worker_runs = {"count": 0}

    def dummy_worker():
        worker_runs["count"] += 1
        sup.heartbeat("worker_1")

    sup.register_thread("worker_1", target=dummy_worker, max_restarts=3)
    sup.start()
    sup.start_thread("worker_1")

    time.sleep(0.2)
    sup.heartbeat("worker_1")
    status = sup.get_status()
    assert "worker_1" in status

    sup.stop()


# 4. HealthMonitor Tests
def test_health_monitor():
    qm = QueueManager()
    qm.create_queue("q1", maxsize=10)
    hm = HealthMonitor(queue_manager=qm)

    hm.record_inference_latency(15.5)
    hm.record_inference_latency(24.5)
    assert hm.get_average_latency_ms() == 20.0

    for _ in range(5):
        hm.record_frame()
        time.sleep(0.01)
    assert hm.get_fps() > 0.0

    snapshot = hm.collect_snapshot()
    assert snapshot.cpu_percent >= 0.0
    assert snapshot.ram_percent >= 0.0
    assert "q1" in snapshot.queue_sizes


# 5. TemporaryStorageService Tests
def test_temporary_storage_service(tmp_path):
    storage = TemporaryStorageService(storage_dir=tmp_path, retention_seconds=0.2, use_memory_backend=False)
    arr = np.zeros((10, 10), dtype=np.float32)

    seg1 = storage.save_segment(segment_id="seg_01", modality="video", data=arr, duration_seconds=0.1)
    assert seg1.filepath is not None and seg1.filepath.exists()

    retrieved = storage.get_segment("seg_01")
    assert retrieved is not None

    time.sleep(0.3)
    pruned = storage.prune_old_segments()
    assert pruned >= 1
    storage.clear()


# 6. HistoryService Tests
def test_history_service(tmp_path):
    hs = HistoryService(max_records=100)
    pred = PredictionResult(
        label=DetectionLabel.FAKE,
        confidence=0.95,
        modality=Modality.FUSED,
        latency_ms=12.0,
    )
    rec = hs.add_prediction(pred, metadata={"source": "cam_0"})
    assert rec.label == "FAKE"

    records = hs.query(label="FAKE", min_confidence=0.90)
    assert len(records) == 1

    json_path = tmp_path / "history.json"
    csv_path = tmp_path / "history.csv"
    hs.export_json(json_path)
    hs.export_csv(csv_path)

    assert json_path.exists()
    assert csv_path.exists()
    hs.clear()
    assert len(hs) == 0


# 7. NotificationService Tests
def test_notification_service():
    ns = NotificationService(fake_alert_threshold=0.8)
    events: list[NotificationEvent] = []

    def subscriber(evt: NotificationEvent):
        events.append(evt)

    ns.subscribe(subscriber)
    evt1 = ns.publish(title="Test", message="Info event", severity=NotificationSeverity.INFO)
    assert len(events) == 1

    alert_evt = ns.notify_fake_detected(confidence=0.92, modality="video")
    assert alert_evt is not None
    assert len(events) == 2
    assert events[-1].severity == NotificationSeverity.ALERT

    ns.unsubscribe(subscriber)


# 8. SessionOrchestrator Tests
def test_session_orchestrator():
    orchestrator = SessionOrchestrator()
    assert orchestrator.state == SessionState.IDLE

    assert orchestrator.start() is True
    assert orchestrator.state == SessionState.RUNNING

    assert orchestrator.pause() is True
    assert orchestrator.state == SessionState.PAUSED

    assert orchestrator.resume() is True
    assert orchestrator.state == SessionState.RUNNING

    summary = orchestrator.get_summary()
    assert summary["state"] == "RUNNING"
    assert "health_snapshot" in summary

    assert orchestrator.stop() is True
    assert orchestrator.state == SessionState.STOPPED
