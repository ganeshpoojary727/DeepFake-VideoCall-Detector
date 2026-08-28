"""Audio processing constants and default hyper-parameters.

Provides central constants for sampling rates, STFT window parameters,
Mel spectrogram dimensions, supported audio extensions, normalization defaults,
and production audio dataset identifiers.
"""

from __future__ import annotations

from typing import Tuple

# Sampling and Audio Length Defaults
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_TARGET_DURATION_SECONDS: float = 5.0
DEFAULT_TARGET_LENGTH_FRAMES: int = 100
DEFAULT_NUM_SAMPLES: int = 64600  # Default ~4 sec @ 16kHz for AASIST

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

# Production Paths
AUDIO_DATASETS_DIR: str = "datasets/audio"
AUDIO_CACHE_DIR: str = "datasets/audio/cache"
AUDIO_LOGS_DIR: str = "logs/audio"
AUDIO_MODELS_DIR: str = "trained_models/audio"

# Production Dataset Identifiers
DATASET_ASVSPOOF_2019_LA: str = "asvspoof2019_la"
DATASET_ASVSPOOF_2021_LA: str = "asvspoof2021_la"
DATASET_ASVSPOOF_2021_DF: str = "asvspoof2021_df"

SUPPORTED_AUDIO_DATASETS: Tuple[str, ...] = (
    DATASET_ASVSPOOF_2019_LA,
    DATASET_ASVSPOOF_2021_LA,
    DATASET_ASVSPOOF_2021_DF,
)

# Target Production Model
PRODUCTION_AUDIO_MODEL: str = "aasist"

# ASVspoof Standard Labels
LABEL_BONAFIDE: int = 0
LABEL_SPOOF: int = 1
LABEL_BONAFIDE_STR: str = "bonafide"
LABEL_SPOOF_STR: str = "spoof"

# Biometric & minDCF Parameter Defaults
DEFAULT_MIN_DCF_P_TARGET: float = 0.05
DEFAULT_MIN_DCF_C_MISS: float = 1.0
DEFAULT_MIN_DCF_C_FA: float = 1.0
