"""
Tests for the new unified MediaAnalyzer architecture:
- MediaRouter (type detection & validation)
- AnalysisReport (data structure and serialization)
- AudioAnalyzer (audio deepfake detection)
- ImageAnalyzer (image deepfake detection)
- VideoAnalyzer (video + audio fusion detection)
- MediaAnalyzer (top-level orchestrator)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
import cv2

from app.analyzer.analysis_report import AnalysisReport
from app.analyzer.media_router import MediaRouter, MediaType
from app.analyzer.media_analyzer import MediaAnalyzer
from app.analyzer.audio_analyzer import AudioAnalyzer
from app.analyzer.image_analyzer import ImageAnalyzer
from app.analyzer.video_analyzer import VideoAnalyzer


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Create a temporary synthetic 16kHz sine wave audio file."""
    wav_path = tmp_path / "sample_test.wav"
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(wav_path), sine, sr)
    return wav_path


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a temporary synthetic RGB image."""
    img_path = tmp_path / "sample_face.jpg"
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, (200, 180, 160), -1)  # skin-toned circle
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a temporary synthetic video file."""
    vid_path = tmp_path / "sample_vid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (224, 224))
    for _ in range(20):
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return vid_path


# ──────────────────────────────────────────────
# 1. MediaRouter Tests
# ──────────────────────────────────────────────

class TestMediaRouter:
    def test_detect_type_image(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            assert MediaRouter.detect_type(f"file{ext}") == MediaType.IMAGE

    def test_detect_type_video(self):
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            assert MediaRouter.detect_type(f"file{ext}") == MediaType.VIDEO

    def test_detect_type_audio(self):
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            assert MediaRouter.detect_type(f"file{ext}") == MediaType.AUDIO

    def test_detect_type_unknown(self):
        assert MediaRouter.detect_type("file.xyz") == MediaType.UNKNOWN
        assert MediaRouter.detect_type("file.txt") == MediaType.UNKNOWN

    def test_validate_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MediaRouter.validate_file("non_existent_file_12345.mp4")

    def test_validate_file_success(self, sample_wav: Path):
        validated = MediaRouter.validate_file(sample_wav)
        assert validated.exists()
        assert validated == sample_wav.resolve()


# ──────────────────────────────────────────────
# 2. AnalysisReport Tests
# ──────────────────────────────────────────────

class TestAnalysisReport:
    def test_analysis_report_to_dict(self):
        report = AnalysisReport(
            verdict="REAL",
            confidence=0.15,
            media_type="audio",
            scores={"audio": 0.15},
            processing_time_ms=125.4,
            metadata={"sample_rate": 16000},
        )
        data = report.to_dict()
        assert data["verdict"] == "REAL"
        assert data["confidence"] == 0.15
        assert data["media_type"] == "audio"
        assert data["scores"]["audio"] == 0.15
        assert data["processing_time_ms"] == 125.4
        assert data["metadata"]["sample_rate"] == 16000

    def test_analysis_report_summary(self):
        report = AnalysisReport(
            verdict="FAKE",
            confidence=0.92,
            media_type="video",
            scores={"video": 0.95, "audio": 0.88, "fused": 0.92},
            processing_time_ms=350.0,
        )
        summary = report.summary
        assert "FAKE" in summary
        assert "0.92" in summary
        assert "video" in summary
        assert report.is_fake is True
        assert report.is_real is False


# ──────────────────────────────────────────────
# 3. AudioAnalyzer Tests
# ──────────────────────────────────────────────

class TestAudioAnalyzer:
    def test_audio_analyzer_initialization(self):
        analyzer = AudioAnalyzer(device=torch.device("cpu"))
        assert analyzer is not None

    def test_audio_analyzer_analyze(self, sample_wav: Path):
        analyzer = AudioAnalyzer(device=torch.device("cpu"))
        report = analyzer.analyze(sample_wav)

        assert isinstance(report, AnalysisReport)
        assert report.media_type == "audio"
        assert report.verdict in {"REAL", "FAKE", "UNCERTAIN"}
        assert 0.0 <= report.confidence <= 1.0
        assert "audio" in report.scores
        assert report.processing_time_ms > 0

    def test_audio_analyzer_buffer(self):
        analyzer = AudioAnalyzer(device=torch.device("cpu"))
        buffer = np.zeros(32000, dtype=np.float32)
        score = analyzer.analyze_buffer(buffer, sr=16000)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────
# 4. ImageAnalyzer Tests
# ──────────────────────────────────────────────

class TestImageAnalyzer:
    def test_image_analyzer_initialization(self):
        analyzer = ImageAnalyzer(device=torch.device("cpu"))
        assert analyzer is not None

    def test_image_analyzer_analyze(self, sample_image: Path):
        analyzer = ImageAnalyzer(device=torch.device("cpu"))
        report = analyzer.analyze(sample_image)

        assert isinstance(report, AnalysisReport)
        assert report.media_type == "image"
        assert report.verdict in {"REAL", "FAKE", "UNCERTAIN"}
        assert 0.0 <= report.confidence <= 1.0
        assert "image" in report.scores
        assert report.processing_time_ms > 0


# ──────────────────────────────────────────────
# 5. VideoAnalyzer Tests
# ──────────────────────────────────────────────

class TestVideoAnalyzer:
    def test_video_analyzer_initialization(self):
        analyzer = VideoAnalyzer(device=torch.device("cpu"))
        assert analyzer is not None

    def test_video_analyzer_analyze(self, sample_video: Path):
        analyzer = VideoAnalyzer(device=torch.device("cpu"))
        report = analyzer.analyze(sample_video)

        assert isinstance(report, AnalysisReport)
        assert report.media_type == "video"
        assert report.verdict in {"REAL", "FAKE", "UNCERTAIN"}
        assert 0.0 <= report.confidence <= 1.0
        assert "video" in report.scores
        assert report.processing_time_ms > 0


# ──────────────────────────────────────────────
# 6. MediaAnalyzer Orchestrator Tests
# ──────────────────────────────────────────────

class TestMediaAnalyzerOrchestrator:
    def test_orchestrator_routes_audio(self, sample_wav: Path):
        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze(sample_wav)
        assert report.media_type == "audio"

    def test_orchestrator_routes_image(self, sample_image: Path):
        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze(sample_image)
        assert report.media_type == "image"

    def test_orchestrator_routes_video(self, sample_video: Path):
        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze(sample_video)
        assert report.media_type == "video"

    def test_orchestrator_analyze_batch(self, sample_wav: Path, sample_image: Path):
        analyzer = MediaAnalyzer(device="cpu")
        reports = analyzer.analyze_batch([sample_wav, sample_image])
        assert len(reports) == 2
        assert reports[0].media_type == "audio"
        assert reports[1].media_type == "image"

    def test_orchestrator_analyze_directory(self, tmp_path: Path, sample_wav: Path):
        analyzer = MediaAnalyzer(device="cpu")
        reports = analyzer.analyze_directory(tmp_path)
        assert len(reports) >= 1
        assert any(r.media_type == "audio" for r in reports)

    def test_orchestrator_get_system_status(self):
        analyzer = MediaAnalyzer(device="cpu")
        status = analyzer.get_system_status()
        assert "device" in status
        assert "cuda_available" in status
        assert "models" in status
