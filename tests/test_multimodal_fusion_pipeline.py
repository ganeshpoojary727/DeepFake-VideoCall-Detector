"""
Comprehensive Verification Suite for Phase 3: Multimodal Fusion Engine,
ConsolidatedForensicReport Schema, Dynamic Score Gating, and Unified /api/v1/analyze API.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from app.analyzer.analysis_report import ConsolidatedForensicReport
from app.analyzer.media_analyzer import MediaAnalyzer
from app.api.server import create_app
from app.fusion.inference.fusion_engine import MultimodalFusion


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def test_client() -> TestClient:
    """FastAPI test client instance."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_media_files(tmp_path: Path) -> Dict[str, Path]:
    """Create sample image, audio, and video files."""
    # 1. Sample WAV Audio (2 seconds @ 16kHz)
    sr = 16000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_path = tmp_path / "sample_audio.wav"
    sf.write(str(audio_path), sine, sr)

    # 2. Sample JPG Image (256x256)
    img = np.full((256, 256, 3), fill_value=120, dtype=np.uint8)
    cv2.circle(img, (128, 128), 50, (200, 180, 160), -1)
    img_path = tmp_path / "sample_image.jpg"
    cv2.imwrite(str(img_path), img)

    # 3. Sample MP4 Video (16 frames @ 30fps)
    video_path = tmp_path / "sample_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (224, 224))
    for i in range(16):
        frame = np.full((224, 224, 3), fill_value=int(100 + i * 5), dtype=np.uint8)
        cv2.circle(frame, (112, 112), 40, (190, 170, 150), -1)
        out.write(frame)
    out.release()

    return {
        "audio": audio_path,
        "image": img_path,
        "video": video_path,
    }


# ── 1. Multimodal Fusion Engine Tests ──────────────────────────────────────────

class TestMultimodalFusionEngine:
    """Test suite for dynamic confidence weighting, gating, and temporal synchronization."""

    def test_fusion_engine_unimodal_audio(self) -> None:
        fusion = MultimodalFusion()
        audio_telemetry = {
            "verdict": "FAKE",
            "confidence": 0.88,
            "raw_scores": {"bonafide_prob": 0.12, "spoof_prob": 0.88},
            "spectral_cues": {"peak_artifact_ranges": []},
            "timeline": [
                {"chunk_index": 0, "start_time_sec": 0.0, "end_time_sec": 4.04, "spoof_prob": 0.88, "verdict": "FAKE"}
            ],
        }

        report = fusion.fuse_multimodal(audio_telemetry=audio_telemetry, visual_telemetry=None)
        assert isinstance(report, ConsolidatedForensicReport)
        assert report.media_type == "AUDIO"
        assert report.verdict == "FAKE"
        assert report.overall_confidence == 0.88
        assert report.modality_breakdown["audio"] is not None
        assert report.modality_breakdown["visual"] is None
        assert len(report.temporal_sync) >= 1

    def test_fusion_engine_unimodal_visual(self) -> None:
        fusion = MultimodalFusion()
        visual_telemetry = {
            "verdict": "REAL",
            "confidence": 0.91,
            "raw_scores": {"real_prob": 0.91, "fake_prob": 0.09},
            "visual_cues": {"ela_discrepancy_score": 0.15, "fft_spectral_anomaly": 0.12, "boundary_inconsistency": 0.18},
            "timeline": [
                {"frame_idx": 0, "timestamp_sec": 0.0, "spoof_prob": 0.09, "is_anomaly": False}
            ],
            "key_artifacts": [],
        }

        report = fusion.fuse_multimodal(audio_telemetry=None, visual_telemetry=visual_telemetry, media_type="IMAGE")
        assert report.media_type == "IMAGE"
        assert report.verdict == "REAL"
        assert report.overall_confidence == 0.91
        assert report.modality_breakdown["visual"] is not None
        assert report.modality_breakdown["audio"] is None

    def test_fusion_engine_anomaly_boost_gating(self) -> None:
        fusion = MultimodalFusion(audio_weight=0.5, video_weight=0.5, anomaly_threshold=0.70)

        # High visual tampering (0.85) vs moderate audio (0.40)
        audio_telemetry = {
            "verdict": "REAL",
            "confidence": 0.60,
            "raw_scores": {"bonafide_prob": 0.60, "spoof_prob": 0.40},
            "spectral_cues": {},
            "timeline": [],
        }
        visual_telemetry = {
            "verdict": "FAKE",
            "confidence": 0.85,
            "raw_scores": {"real_prob": 0.15, "fake_prob": 0.85},
            "visual_cues": {"ela_discrepancy_score": 0.85, "fft_spectral_anomaly": 0.80, "boundary_inconsistency": 0.75},
            "timeline": [
                {"frame_idx": 0, "timestamp_sec": 0.0, "spoof_prob": 0.85, "is_anomaly": True}
            ],
            "key_artifacts": [],
        }

        report = fusion.fuse_multimodal(audio_telemetry=audio_telemetry, visual_telemetry=visual_telemetry)
        assert report.media_type == "MULTIMODAL"
        assert report.verdict == "FAKE"
        assert report.overall_confidence >= 0.70  # Anomaly boost applied
        assert report.metadata.get("anomaly_boost_applied") is True


# ── 2. MediaAnalyzer Consolidated Orchestration Tests ──────────────────────────

class TestMediaAnalyzerConsolidated:
    """Test suite for MediaAnalyzer.analyze_consolidated across all media formats."""

    def test_consolidated_audio_analysis(self, sample_media_files: Dict[str, Path]) -> None:
        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze_consolidated(sample_media_files["audio"])

        assert isinstance(report, ConsolidatedForensicReport)
        assert report.media_type == "AUDIO"
        assert report.verdict in ("REAL", "FAKE")
        assert 0.0 <= report.overall_confidence <= 1.0
        assert report.modality_breakdown["audio"] is not None
        assert report.modality_breakdown["visual"] is None
        assert isinstance(report.temporal_sync, list)

    def test_consolidated_image_analysis(self, sample_media_files: Dict[str, Path]) -> None:
        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze_consolidated(sample_media_files["image"])

        assert isinstance(report, ConsolidatedForensicReport)
        assert report.media_type == "IMAGE"
        assert report.verdict in ("REAL", "FAKE")
        assert 0.0 <= report.overall_confidence <= 1.0
        assert report.modality_breakdown["visual"] is not None
        assert report.modality_breakdown["audio"] is None

    def test_consolidated_video_analysis(self, sample_media_files: Dict[str, Path]) -> None:
        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze_consolidated(sample_media_files["video"])

        assert isinstance(report, ConsolidatedForensicReport)
        assert report.media_type in ("VIDEO", "MULTIMODAL")
        assert report.verdict in ("REAL", "FAKE")
        assert 0.0 <= report.overall_confidence <= 1.0
        assert report.modality_breakdown["visual"] is not None
        assert isinstance(report.temporal_sync, list)
        assert len(report.temporal_sync) >= 1


# ── 3. FastAPI /api/v1/analyze Endpoint Tests ──────────────────────────────────

class TestUnifiedAPIEndpoint:
    """Test suite for FastAPI /api/v1/analyze multipart file upload and response schema."""

    def test_api_v1_analyze_audio(self, test_client: TestClient, sample_media_files: Dict[str, Path]) -> None:
        with open(sample_media_files["audio"], "rb") as fh:
            response = test_client.post(
                "/api/v1/analyze",
                files={"file": ("test_audio.wav", fh, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["media_type"] == "AUDIO"
        assert data["verdict"] in ("REAL", "FAKE")
        assert 0.0 <= data["overall_confidence"] <= 1.0
        assert "modality_breakdown" in data
        assert "temporal_sync" in data
        assert "top_anomalies" in data
        assert data["processing_time_ms"] > 0

    def test_api_v1_analyze_image(self, test_client: TestClient, sample_media_files: Dict[str, Path]) -> None:
        with open(sample_media_files["image"], "rb") as fh:
            response = test_client.post(
                "/api/v1/analyze",
                files={"file": ("test_image.jpg", fh, "image/jpeg")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["media_type"] == "IMAGE"
        assert data["verdict"] in ("REAL", "FAKE")
        assert 0.0 <= data["overall_confidence"] <= 1.0
        assert data["modality_breakdown"]["visual"] is not None

    def test_api_v1_analyze_video(self, test_client: TestClient, sample_media_files: Dict[str, Path]) -> None:
        with open(sample_media_files["video"], "rb") as fh:
            response = test_client.post(
                "/api/v1/analyze",
                files={"file": ("test_video.mp4", fh, "video/mp4")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["media_type"] in ("VIDEO", "MULTIMODAL")
        assert data["verdict"] in ("REAL", "FAKE")
        assert 0.0 <= data["overall_confidence"] <= 1.0
        assert len(data["temporal_sync"]) >= 1

    def test_api_v1_analyze_unsupported(self, test_client: TestClient) -> None:
        response = test_client.post(
            "/api/v1/analyze",
            files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]
