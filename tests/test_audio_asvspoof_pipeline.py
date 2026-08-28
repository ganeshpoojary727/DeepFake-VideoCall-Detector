"""
Comprehensive Verification Suite for ASVspoof Standardization, EER/min t-DCF Calibration,
and AASIST Structured Telemetry.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import soundfile as sf
import torch

from app.analyzer.audio_analyzer import AudioAnalyzer
from app.audio.core.base_dataset import BaseAudioDataset
from app.audio.datasets.asvspoof2019 import ASVspoof2019Dataset
from app.audio.datasets.asvspoof2021 import ASVspoof2021Dataset
from app.audio.datasets.audio_dataset import AudioDataset
from app.audio.evaluation.eer import (
    compute_det_curve,
    compute_eer,
    compute_eer_from_labels,
    compute_min_dcf,
    compute_min_tdcf,
    evaluate_cm_predictions,
)
from app.audio.inference.voice_detector import VoiceDetector
from app.audio.models.aasist import AASIST
from app.audio.preprocessing.audio_loader import AudioLoader


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock ASVspoof audio and protocol files."""
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    # Generate 3 mock audio files (2s each)
    for fname in ["LA_T_1000001", "LA_T_1000002", "DF_E_2000001"]:
        # Synthesize sinusoidal signal
        t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(audio_dir / f"{fname}.flac"), audio, sr)

    # ASVspoof 2019 LA format protocol
    p19 = tmp_path / "protocol_2019.txt"
    p19.write_text(
        "LA_0001 LA_T_1000001 - - bonafide\n"
        "LA_0001 LA_T_1000002 - A01 spoof\n",
        encoding="utf-8",
    )

    # ASVspoof 2021 DF format protocol
    p21 = tmp_path / "protocol_2021_df.txt"
    p21.write_text(
        "DF_E_2000001 mp3 eval - bonafide\n",
        encoding="utf-8",
    )

    return tmp_path


# ── 1. Dataset Pipeline Standardization Tests ──────────────────────────────────

class TestDatasetPipelineStandardization:
    """Test suite for AudioLoader, ASVspoof protocols, and BaseAudioDataset."""

    def test_audio_loader_load_and_normalize(self, temp_audio_dir: Path) -> None:
        loader = AudioLoader(target_sr=16000, target_samples=64600)
        audio_file = temp_audio_dir / "flac" / "LA_T_1000001.flac"

        audio, sr = loader.load_audio(audio_file)
        assert sr == 16000
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert np.max(np.abs(audio)) <= 1.0 + 1e-5

    def test_audio_loader_pad_crop_modes(self) -> None:
        loader = AudioLoader(target_sr=16000, target_samples=64600)
        short_signal = np.sin(np.linspace(0, 10, 16000, dtype=np.float32))

        # Wrap padding
        padded_wrap = loader.pad_crop_waveform(short_signal, target_samples=64600, mode="wrap")
        assert len(padded_wrap) == 64600
        assert not np.all(padded_wrap[16000:] == 0.0)

        # Zero padding
        padded_zero = loader.pad_crop_waveform(short_signal, target_samples=64600, mode="zero")
        assert len(padded_zero) == 64600
        assert np.all(padded_zero[16000:] == 0.0)

        # Long signal crop
        long_signal = np.ones(100000, dtype=np.float32)
        cropped = loader.pad_crop_waveform(long_signal, target_samples=64600)
        assert len(cropped) == 64600

    def test_audio_loader_chunking(self) -> None:
        loader = AudioLoader(target_sr=16000, target_samples=64600)
        # 10 second audio (160,000 samples)
        long_audio = np.random.randn(160000).astype(np.float32)
        chunks = loader.chunk_waveform(long_audio, chunk_samples=64600, hop_samples=32300)

        assert len(chunks) > 1
        for chunk, start_sec, end_sec in chunks:
            assert len(chunk) == 64600
            assert start_sec < end_sec

    def test_asvspoof_2019_protocol_parsing(self, temp_audio_dir: Path) -> None:
        p19 = temp_audio_dir / "protocol_2019.txt"
        records = AudioLoader.parse_asvspoof_protocol(p19)

        assert len(records) == 2
        assert records[0]["file_name"] == "LA_T_1000001"
        assert records[0]["label"] == 0
        assert records[0]["label_str"] == "bonafide"

        assert records[1]["file_name"] == "LA_T_1000002"
        assert records[1]["label"] == 1
        assert records[1]["label_str"] == "spoof"

    def test_asvspoof_2021_protocol_parsing(self, temp_audio_dir: Path) -> None:
        p21 = temp_audio_dir / "protocol_2021_df.txt"
        records = AudioLoader.parse_asvspoof_protocol(p21)

        assert len(records) == 1
        assert records[0]["file_name"] == "DF_E_2000001"
        assert records[0]["label"] == 0
        assert records[0]["label_str"] == "bonafide"

    def test_asvspoof_dataset_classes(self, temp_audio_dir: Path) -> None:
        audio_dir = temp_audio_dir / "flac"
        p19 = temp_audio_dir / "protocol_2019.txt"

        ds19 = ASVspoof2019Dataset(protocol_file=p19, audio_directory=audio_dir)
        assert len(ds19) == 2
        assert isinstance(ds19, BaseAudioDataset)

        dist = ds19.get_label_distribution()
        assert dist[0] == 1  # 1 bonafide
        assert dist[1] == 1  # 1 spoof

        # Test __getitem__
        tensor, label = ds19[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (64600,)
        assert label == 0


# ── 2. Metric & Evaluation Calibration Tests ───────────────────────────────────

class TestMetricEvaluationCalibration:
    """Test suite for EER, min t-DCF, min DCF, and biometric error rates."""

    def test_compute_eer_separated_distributions(self) -> None:
        # Well-separated bona fide and spoof scores
        bonafide_scores = np.random.normal(loc=1.0, scale=0.1, size=100)
        spoof_scores = np.random.normal(loc=-1.0, scale=0.1, size=100)

        eer, threshold = compute_eer(bonafide_scores, spoof_scores)
        assert eer is not None
        assert 0.0 <= eer <= 0.05
        assert -1.0 <= threshold <= 1.0

    def test_compute_eer_overlapping_distributions(self) -> None:
        # Overlapping distributions -> EER ~ 0.5
        bonafide_scores = np.random.normal(loc=0.0, scale=1.0, size=500)
        spoof_scores = np.random.normal(loc=0.0, scale=1.0, size=500)

        eer, threshold = compute_eer(bonafide_scores, spoof_scores)
        assert eer is not None
        assert 0.40 <= eer <= 0.60

    def test_compute_min_dcf(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.85, 0.9, 0.8, 0.95])

        min_dcf = compute_min_dcf(y_true, y_scores, p_target=0.05)
        assert isinstance(min_dcf, float)
        assert 0.0 <= min_dcf <= 1.0
        assert min_dcf == 0.0  # Perfect separation

    def test_evaluate_cm_predictions_suite(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.15, 0.25, 0.85, 0.9, 0.8, 0.95])

        metrics = evaluate_cm_predictions(y_true, y_scores)
        assert "eer" in metrics
        assert "min_tdcf" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "apcer" in metrics
        assert "bpcer" in metrics
        assert metrics["accuracy"] == 1.0
        assert metrics["auc"] == 1.0


# ── 3. Structured Output & Diagnostic Telemetry Tests ──────────────────────────

class TestStructuredTelemetryInference:
    """Test suite asserting strict adherence to the structured output schema."""

    def test_aasist_forward_shape(self) -> None:
        model = AASIST(num_classes=2)
        model.eval()
        dummy_input = torch.randn(2, 64600)

        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (2, 2)

    def test_voice_detector_structured_schema(self, temp_audio_dir: Path) -> None:
        audio_file = temp_audio_dir / "flac" / "LA_T_1000001.flac"
        detector = VoiceDetector()

        result = detector.predict_file(audio_file)

        # 1. Assert top-level keys
        assert "verdict" in result, "Missing 'verdict' key"
        assert "confidence" in result, "Missing 'confidence' key"
        assert "raw_scores" in result, "Missing 'raw_scores' key"
        assert "spectral_cues" in result, "Missing 'spectral_cues' key"
        assert "timeline" in result, "Missing 'timeline' key"

        # 2. Assert types and ranges
        assert result["verdict"] in ("REAL", "FAKE")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

        # 3. Assert raw_scores schema
        raw = result["raw_scores"]
        assert "bonafide_prob" in raw
        assert "spoof_prob" in raw
        assert 0.0 <= raw["bonafide_prob"] <= 1.0
        assert 0.0 <= raw["spoof_prob"] <= 1.0
        assert pytest.approx(raw["bonafide_prob"] + raw["spoof_prob"], abs=1e-3) == 1.0

        # 4. Assert spectral_cues schema
        cues = result["spectral_cues"]
        assert "peak_artifact_ranges" in cues
        assert "spectral_rolloff_hz" in cues
        assert "high_freq_energy_ratio" in cues
        assert isinstance(cues["peak_artifact_ranges"], list)
        assert isinstance(cues["spectral_rolloff_hz"], float)
        assert isinstance(cues["high_freq_energy_ratio"], float)

        # 5. Assert timeline schema
        timeline = result["timeline"]
        assert isinstance(timeline, list)
        assert len(timeline) >= 1
        for chunk in timeline:
            assert "chunk_index" in chunk
            assert "start_time_sec" in chunk
            assert "end_time_sec" in chunk
            assert "spoof_prob" in chunk
            assert "bonafide_prob" in chunk
            assert "verdict" in chunk
            assert chunk["verdict"] in ("REAL", "FAKE")
            assert 0.0 <= chunk["spoof_prob"] <= 1.0

    def test_audio_analyzer_structured_schema(self, temp_audio_dir: Path) -> None:
        audio_file = temp_audio_dir / "flac" / "LA_T_1000001.flac"
        analyzer = AudioAnalyzer()

        # Structured direct method
        result = analyzer.analyze_structured(audio_file)
        assert result["verdict"] in ("REAL", "FAKE")
        assert "raw_scores" in result
        assert "spectral_cues" in result
        assert "timeline" in result

        # Unified report analysis
        report = analyzer.analyze(audio_file)
        assert report.verdict in ("REAL", "FAKE", "UNCERTAIN")
        assert "spectral_cues" in report.metadata
        assert "timeline" in report.metadata
        assert "raw_scores" in report.metadata
