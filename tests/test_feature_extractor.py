"""Tests for the FeatureExtractor."""

from __future__ import annotations

import numpy as np
import torch

from app.audio.features.feature_extractor import FeatureExtractor
from app.config.settings import settings


class TestFeatureExtractor:
    """Test suite for feature extraction."""

    def test_extract_output_shape(
        self, extractor: FeatureExtractor, sample_audio: np.ndarray
    ) -> None:
        """Extracted features should have shape (1, n_mels, target_length)."""
        feature = extractor.extract(sample_audio)
        expected = (1, settings.audio.n_mels, settings.audio.target_length)
        assert feature.shape == expected, f"Expected {expected}, got {feature.shape}"

    def test_extract_output_type(
        self, extractor: FeatureExtractor, sample_audio: np.ndarray
    ) -> None:
        """Extracted features should be float32 tensors."""
        feature = extractor.extract(sample_audio)
        assert isinstance(feature, torch.Tensor)
        assert feature.dtype == torch.float32

    def test_normalization_applied(
        self, extractor: FeatureExtractor, sample_audio: np.ndarray
    ) -> None:
        """After normalization, the spectrogram should be roughly zero-mean."""
        mel = extractor.create_mel_spectrogram(sample_audio)
        mel_db = extractor.convert_to_db(mel)
        normalized = extractor.normalize_spectrogram(mel_db)
        assert abs(normalized.mean()) < 0.1, "Normalized mean should be ~0"

    def test_resize_pads_short(self, extractor: FeatureExtractor) -> None:
        """Short spectrograms should be padded to target_length."""
        short = np.random.randn(128, 50).astype(np.float32)
        resized = extractor.resize_spectrogram(short)
        assert resized.shape[1] == settings.audio.target_length

    def test_resize_crops_long(self, extractor: FeatureExtractor) -> None:
        """Long spectrograms should be cropped to target_length."""
        long_spec = np.random.randn(128, 200).astype(np.float32)
        resized = extractor.resize_spectrogram(long_spec)
        assert resized.shape[1] == settings.audio.target_length

    def test_config_driven_params(self) -> None:
        """Extractor should read parameters from Settings."""
        extractor = FeatureExtractor()
        assert extractor.sample_rate == settings.audio.sample_rate
        assert extractor.n_fft == settings.audio.n_fft
        assert extractor.hop_length == settings.audio.hop_length
        assert extractor.n_mels == settings.audio.n_mels
        assert extractor.target_length == settings.audio.target_length

    def test_custom_params(self) -> None:
        """Extractor should accept custom parameters."""
        extractor = FeatureExtractor(n_mels=64, target_length=50)
        assert extractor.n_mels == 64
        assert extractor.target_length == 50