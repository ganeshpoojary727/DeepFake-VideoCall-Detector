"""
Pytest fixtures shared across all test modules.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.audio.models.cnn_model import DeepFakeCNN
from app.audio.preprocessing.audio_preprocessor import AudioPreprocessor
from app.audio.features.feature_extractor import FeatureExtractor
from app.config.settings import settings


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a synthetic 1-second audio waveform at 16 kHz."""
    sr = settings.audio.sample_rate
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    # Mix of tones to simulate speech-like content
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    return audio.astype(np.float32)


@pytest.fixture
def preprocessor() -> AudioPreprocessor:
    """Return a fresh AudioPreprocessor."""
    return AudioPreprocessor()


@pytest.fixture
def extractor() -> FeatureExtractor:
    """Return a fresh FeatureExtractor (no augmentation)."""
    return FeatureExtractor(apply_augmentation=False)


@pytest.fixture
def model() -> DeepFakeCNN:
    """Return an untrained model on CPU."""
    return DeepFakeCNN(num_classes=2)


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    """Return a dummy batch tensor matching expected input shape."""
    return torch.randn(
        4, 1, settings.audio.n_mels, settings.audio.target_length
    )
