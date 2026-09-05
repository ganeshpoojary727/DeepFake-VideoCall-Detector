"""Unit tests for RealtimeLiveDetector and live streaming primitives."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.realtime.stream_detector import RealtimeLiveDetector


@pytest.fixture
def detector() -> RealtimeLiveDetector:
    """Fixture providing initialized RealtimeLiveDetector on CPU."""
    return RealtimeLiveDetector(device=torch.device("cpu"), window_size=4)


def test_detector_initialization(detector: RealtimeLiveDetector):
    """Verify detector components initialize properly."""
    assert detector is not None
    assert detector.window_size == 4
    assert detector.smoothed_fake_prob == 0.5
    assert detector.current_verdict == "UNCERTAIN"


def test_process_black_frame(detector: RealtimeLiveDetector):
    """Verify non-biometric/black frame triggers Stage-0 guard correctly."""
    black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = detector.process_frame(black_frame)

    assert "verdict" in res
    assert "confidence" in res
    assert "fps" in res
    assert "latency_ms" in res
    assert res["verdict"] in ("NOT_APPLICABLE", "UNCERTAIN")


def test_process_base64_frame(detector: RealtimeLiveDetector):
    """Verify base64 frame encoding/decoding pipeline."""
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buffer).decode("utf-8")

    res = detector.process_base64_frame(b64_str)
    assert "verdict" in res
    assert "status" in res


def test_detector_reset(detector: RealtimeLiveDetector):
    """Verify reset properly clears buffer and tracking history."""
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    detector.process_frame(img)
    detector.process_frame(img)
    assert detector.frame_count >= 2

    detector.reset()
    assert detector.frame_count == 0
    assert len(detector.history_points) == 0
    assert detector.smoothed_fake_prob == 0.5
