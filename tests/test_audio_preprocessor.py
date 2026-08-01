"""Tests for the AudioPreprocessor."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.preprocessing.audio_preprocessor import AudioPreprocessor


class TestAudioPreprocessor:
    """Test suite for audio preprocessing."""

    def test_normalize_audio(self, preprocessor: AudioPreprocessor) -> None:
        """Normalization should produce values in [-1, 1]."""
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        normalized = preprocessor.normalize_audio(audio)
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-6

    def test_trim_silence(self, preprocessor: AudioPreprocessor) -> None:
        """Trimming should reduce length when silence is present."""
        # Create audio with silence padding
        signal = np.concatenate([
            np.zeros(4000, dtype=np.float32),
            np.random.randn(8000).astype(np.float32) * 0.5,
            np.zeros(4000, dtype=np.float32),
        ])
        trimmed = preprocessor.trim_silence(signal)
        assert len(trimmed) < len(signal), "Trimming should remove silence"

    def test_pipeline_order(self, preprocessor: AudioPreprocessor, sample_audio: np.ndarray) -> None:
        """Pipeline should produce a valid waveform."""
        # We can't test the full pipeline without a file, but we can test
        # that trim → normalize produces valid output
        trimmed = preprocessor.trim_silence(sample_audio)
        normalized = preprocessor.normalize_audio(trimmed)
        assert len(normalized) > 0
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-6

    def test_load_nonexistent_file(self, preprocessor: AudioPreprocessor) -> None:
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_audio("/nonexistent/path/audio.wav")

    def test_sample_rate(self, preprocessor: AudioPreprocessor) -> None:
        """Preprocessor should use the configured sample rate."""
        assert preprocessor.sample_rate == 16000