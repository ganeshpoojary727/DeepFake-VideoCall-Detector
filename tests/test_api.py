"""
Tests for the FastAPI REST API Server (app/api/server.py).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.api.server import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes() -> bytes:
    buf = io.BytesIO()
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(buf, sine, sr, format="WAV")
    return buf.getvalue()


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "supported_extensions" in data


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "models" in data


def test_detect_file_endpoint(client: TestClient, sample_wav_bytes: bytes):
    response = client.post(
        "/detect/file",
        files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in {"REAL", "FAKE", "UNCERTAIN"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["media_type"] == "audio"
    assert "audio" in data["scores"]


def test_detect_file_unsupported_format(client: TestClient):
    response = client.post(
        "/detect/file",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_detect_batch_endpoint(client: TestClient, sample_wav_bytes: bytes):
    response = client.post(
        "/detect/batch",
        files=[
            ("files", ("sample1.wav", sample_wav_bytes, "audio/wav")),
            ("files", ("sample2.wav", sample_wav_bytes, "audio/wav")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["verdict"] in {"REAL", "FAKE", "UNCERTAIN"}
