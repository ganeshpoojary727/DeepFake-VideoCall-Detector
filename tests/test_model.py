"""Tests for configuration, logger, helpers, and core interfaces."""

from __future__ import annotations

import numpy as np
import pytest

from app.config.settings import Settings, settings
from app.core.interfaces import DetectionLabel, EvaluationResult, Modality, PredictionResult
from app.utils.helpers import set_seed, validate_audio_file
from app.utils.logger import get_logger


class TestSettings:
    """Test suite for Settings configuration."""

    def test_singleton_exists(self) -> None:
        """Module-level settings singleton should be a Settings instance."""
        assert isinstance(settings, Settings)

    def test_log_dir_exists(self) -> None:
        """LOG_DIR attribute should exist (fixes audit bug)."""
        assert hasattr(settings, "LOG_DIR")
        assert settings.LOG_DIR is not None

    def test_lazy_device(self) -> None:
        """DEVICE property should return a torch.device."""
        import torch
        assert isinstance(settings.DEVICE, torch.device)

    def test_audio_config(self) -> None:
        """Audio sub-config should have expected defaults."""
        assert settings.audio.sample_rate == 16000
        assert settings.audio.n_mels == 128
        assert settings.audio.n_fft == 2048
        assert settings.audio.hop_length == 512

    def test_backward_compat_aliases(self) -> None:
        """Backward-compatible aliases should match sub-config values."""
        assert settings.SAMPLE_RATE == settings.audio.sample_rate
        assert settings.N_MELS == settings.audio.n_mels
        assert settings.BATCH_SIZE == settings.training.batch_size

    def test_validate(self) -> None:
        """Default settings should pass validation."""
        warnings = settings.validate()
        assert isinstance(warnings, list)
        assert len(warnings) == 0


class TestLogger:
    """Test suite for the logging module."""

    def test_get_logger(self) -> None:
        """get_logger should return a Logger instance."""
        import logging
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)
        assert "DeepFakeDetector" in log.name

    def test_logger_has_handlers(self) -> None:
        """Root logger should have at least one handler."""
        import logging
        root = logging.getLogger("DeepFakeDetector")
        assert len(root.handlers) > 0


class TestHelpers:
    """Test suite for utility helpers."""

    def test_validate_nonexistent_file(self) -> None:
        """validate_audio_file should raise for missing files."""
        with pytest.raises(FileNotFoundError):
            validate_audio_file("/nonexistent/audio.wav")

    def test_set_seed_runs(self) -> None:
        """set_seed should not raise."""
        set_seed(42)

    def test_set_seed_reproducibility(self) -> None:
        """Same seed should produce same random values."""
        set_seed(123)
        a = np.random.rand(10)
        set_seed(123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)


class TestCoreInterfaces:
    """Test suite for core domain objects."""

    def test_prediction_result_valid(self) -> None:
        """PredictionResult should accept valid confidence."""
        result = PredictionResult(
            label=DetectionLabel.REAL,
            confidence=0.85,
            modality=Modality.AUDIO,
        )
        assert result.label == DetectionLabel.REAL
        assert result.confidence == 0.85

    def test_prediction_result_invalid_confidence(self) -> None:
        """PredictionResult should reject confidence outside [0, 1]."""
        with pytest.raises(ValueError):
            PredictionResult(
                label=DetectionLabel.FAKE,
                confidence=1.5,
                modality=Modality.AUDIO,
            )

    def test_detection_label_values(self) -> None:
        """DetectionLabel should have REAL, FAKE, UNCERTAIN."""
        assert DetectionLabel.REAL.value == "REAL"
        assert DetectionLabel.FAKE.value == "FAKE"
        assert DetectionLabel.UNCERTAIN.value == "UNCERTAIN"