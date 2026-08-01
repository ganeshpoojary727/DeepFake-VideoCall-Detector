"""Audio processing constants and default hyper-parameters.

Provides central constants for sampling rates, STFT window parameters,
Mel spectrogram dimensions, supported audio extensions, and normalization defaults.
"""

from __future__ import annotations

from typing import Tuple

# Sampling and Audio Length Defaults
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_TARGET_DURATION_SECONDS: float = 5.0
DEFAULT_TARGET_LENGTH_FRAMES: int = 100

# STFT and Spectrogram Parameter Defaults
DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_WIN_LENGTH: int = 2048
DEFAULT_N_MELS: int = 128

# Frequency Bounds
DEFAULT_F_MIN: float = 0.0
DEFAULT_F_MAX: float = 8000.0

# Audio File Format Extensions
SUPPORTED_AUDIO_EXTENSIONS: Tuple[str, ...] = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
)

# Audio Preprocessing Defaults
SILENCE_TOP_DB: int = 30
PEAK_NORM_TARGET_DB: float = -3.0
EPSILON: float = 1e-8
