"""
Comprehensive Verification Suite for Phase 4: Generative Forensic Explainer,
Dual-Engine Natural Language Report Synthesis, Telemetry Grounding, and Export Utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.analyzer.analysis_report import ConsolidatedForensicReport
from app.analyzer.forensic_explainer import (
    DeterministicExplainerProvider,
    GenerativeForensicExplainer,
    NaturalLanguageReport,
    export_json_certificate,
    export_markdown_report,
)
from app.analyzer.media_analyzer import MediaAnalyzer
from app.api.server import create_app


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def test_client() -> TestClient:
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_multimodal_report() -> ConsolidatedForensicReport:
    """Fixture providing a mock multimodal ConsolidatedForensicReport."""
    return ConsolidatedForensicReport(
        media_type="MULTIMODAL",
        verdict="FAKE",
        overall_confidence=0.942,
        modality_breakdown={
            "audio": {
                "verdict": "FAKE",
                "confidence": 0.92,
                "raw_scores": {"bonafide_prob": 0.08, "spoof_prob": 0.92},
                "spectral_cues": {
                    "spectral_rolloff_hz": 5800.0,
                    "spectral_flatness": 0.0345,
                    "high_freq_energy_ratio": 0.012,
                    "artifacts_detected": ["High-frequency truncation above 6kHz"],
                },
                "timeline": [{"chunk_index": 0, "start_time_sec": 0.0, "end_time_sec": 4.0, "spoof_prob": 0.92}],
            },
            "visual": {
                "verdict": "FAKE",
                "confidence": 0.95,
                "raw_scores": {"real_prob": 0.05, "fake_prob": 0.95},
                "visual_cues": {
                    "ela_discrepancy_score": 0.78,
                    "fft_spectral_anomaly": 0.82,
                    "boundary_inconsistency": 0.74,
                },
                "timeline": [{"frame_idx": 0, "timestamp_sec": 0.0, "spoof_prob": 0.95, "is_anomaly": True}],
                "key_artifacts": [
                    {"frame_idx": 0, "timestamp_sec": 0.0, "bbox": [10, 10, 100, 100], "spoof_prob": 0.95, "saliency_peak": [50, 50]}
                ],
            },
            "classical_forensics": {
                "ela_discrepancy_score": 0.78,
                "fft_spectral_anomaly": 0.82,
                "boundary_inconsistency": 0.74,
            },
        },
        temporal_sync=[
            {"second": 0.0, "audio_spoof_prob": 0.92, "visual_spoof_prob": 0.95, "fused_spoof_prob": 0.942, "is_anomaly": True},
            {"second": 1.0, "audio_spoof_prob": 0.88, "visual_spoof_prob": 0.91, "fused_spoof_prob": 0.898, "is_anomaly": True},
        ],
        top_anomalies=[
            {"timestamp_sec": 0.0, "modality": "cross_modal", "description": "Synchronized multimodal anomaly peak", "anomaly_score": 0.942}
        ],
        processing_time_ms=125.4,
        metadata={"file_name": "interview_tampered.mp4"},
    )


@pytest.fixture
def sample_authentic_audio_report() -> ConsolidatedForensicReport:
    """Fixture providing a mock authentic audio report."""
    return ConsolidatedForensicReport(
        media_type="AUDIO",
        verdict="REAL",
        overall_confidence=0.985,
        modality_breakdown={
            "audio": {
                "verdict": "REAL",
                "confidence": 0.985,
                "raw_scores": {"bonafide_prob": 0.985, "spoof_prob": 0.015},
                "spectral_cues": {
                    "spectral_rolloff_hz": 7800.0,
                    "spectral_flatness": 0.085,
                    "high_freq_energy_ratio": 0.045,
                    "artifacts_detected": [],
                },
                "timeline": [{"chunk_index": 0, "start_time_sec": 0.0, "end_time_sec": 4.0, "spoof_prob": 0.015}],
            },
            "visual": None,
            "classical_forensics": {},
        },
        temporal_sync=[
            {"second": 0.0, "audio_spoof_prob": 0.015, "visual_spoof_prob": None, "fused_spoof_prob": 0.015, "is_anomaly": False}
        ],
        top_anomalies=[],
        processing_time_ms=64.2,
        metadata={"file_name": "genuine_voice.wav"},
    )


# ── 1. Generative Forensic Explainer Engine Tests ──────────────────────────────

class TestGenerativeForensicExplainer:
    """Test suite for NaturalLanguageReport synthesis, telemetry grounding, and fallback."""

    def test_multimodal_grounded_synthesis(self, sample_multimodal_report: ConsolidatedForensicReport) -> None:
        explainer = GenerativeForensicExplainer()
        nl_report = explainer.explain(sample_multimodal_report, force_provider="deterministic")

        assert isinstance(nl_report, NaturalLanguageReport)
        assert nl_report.provider_used == "deterministic_rules"

        # 1. Executive summary assertions
        assert "FAKE" in nl_report.executive_summary
        assert "94.2%" in nl_report.executive_summary

        # 2. Visual narrative assertions
        assert "0.78" in nl_report.visual_analysis_narrative or "Error Level Analysis" in nl_report.visual_analysis_narrative
        assert "0.82" in nl_report.visual_analysis_narrative or "Fourier" in nl_report.visual_analysis_narrative

        # 3. Audio narrative assertions
        assert "5800" in nl_report.audio_analysis_narrative or "truncation" in nl_report.audio_analysis_narrative

        # 4. Temporal notes assertions
        assert "0.00s" in nl_report.temporal_inconsistency_notes or "anomaly" in nl_report.temporal_inconsistency_notes

        # 5. Recommendations assertions
        assert len(nl_report.forensic_recommendations) >= 2

    def test_authentic_audio_grounded_synthesis(self, sample_authentic_audio_report: ConsolidatedForensicReport) -> None:
        explainer = GenerativeForensicExplainer()
        nl_report = explainer.explain(sample_authentic_audio_report, force_provider="deterministic")

        assert "AUTHENTIC" in nl_report.executive_summary or "REAL" in nl_report.executive_summary
        assert "98.5%" in nl_report.executive_summary
        assert "not conducted" in nl_report.visual_analysis_narrative
        assert "7800" in nl_report.audio_analysis_narrative or "harmonics" in nl_report.audio_analysis_narrative


# ── 2. Export Utilities Tests ──────────────────────────────────────────────────

class TestExportUtilities:
    """Test suite for Markdown audit export and ISO/IEC JSON forensic certificate."""

    def test_markdown_report_export(self, sample_multimodal_report: ConsolidatedForensicReport) -> None:
        explainer = GenerativeForensicExplainer()
        sample_multimodal_report.natural_language_report = explainer.explain(sample_multimodal_report).to_dict()

        md = export_markdown_report(sample_multimodal_report)
        assert isinstance(md, str)
        assert "# Digital Forensics Investigation Report" in md
        assert "DEEPFAKE DETECTED" in md
        assert "94.2%" in md
        assert "## 1. Executive Summary" in md
        assert "## 3. Synchronized Timeline Evidence" in md
        assert "## 4. Forensic Recommendations" in md

    def test_json_certificate_export(self, sample_multimodal_report: ConsolidatedForensicReport) -> None:
        explainer = GenerativeForensicExplainer()
        sample_multimodal_report.natural_language_report = explainer.explain(sample_multimodal_report).to_dict()

        cert = export_json_certificate(sample_multimodal_report)
        assert isinstance(cert, dict)
        assert "certificate_id" in cert
        assert cert["certificate_id"].startswith("CERT-")
        assert "integrity_sha256" in cert
        assert len(cert["integrity_sha256"]) == 64  # Valid sha256 hex string
        assert cert["investigation_results"]["verdict"] == "FAKE"
        assert cert["investigation_results"]["overall_confidence"] == 0.942


# ── 3. MediaAnalyzer & API Endpoint Integration Tests ──────────────────────────

class TestMediaAnalyzerAndAPIIntegration:
    """Test suite for end-to-end media analysis and FastAPI export endpoints."""

    def test_media_analyzer_populates_natural_language_report(self, tmp_path: Path) -> None:
        # Create small test wav file
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav_path = tmp_path / "voice_test.wav"
        sf.write(str(wav_path), sine, sr)

        analyzer = MediaAnalyzer(device="cpu")
        report = analyzer.analyze_consolidated(wav_path)

        assert report.natural_language_report is not None
        assert "executive_summary" in report.natural_language_report
        assert "audio_analysis_narrative" in report.natural_language_report

    def test_api_export_markdown_endpoint(self, test_client: TestClient, tmp_path: Path) -> None:
        img_path = tmp_path / "test_exp.jpg"
        img = np.full((128, 128, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        with open(img_path, "rb") as fh:
            resp = test_client.post(
                "/api/v1/export/markdown",
                files={"file": ("test_exp.jpg", fh, "image/jpeg")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "markdown_report" in data
        assert "# Digital Forensics Investigation Report" in data["markdown_report"]

    def test_api_export_certificate_endpoint(self, test_client: TestClient, tmp_path: Path) -> None:
        img_path = tmp_path / "test_cert.jpg"
        img = np.full((128, 128, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        with open(img_path, "rb") as fh:
            resp = test_client.post(
                "/api/v1/export/certificate",
                files={"file": ("test_cert.jpg", fh, "image/jpeg")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "certificate_id" in data
        assert "integrity_sha256" in data
        assert data["schema_version"] == "3.0.0-forensics"
