"""
Application-wide configuration.

This file contains all configurable values used throughout
the DeepFake Video Call Detector application.
"""

from pathlib import Path
import torch


class Settings:
    """Central application settings."""

    # -----------------------------
    # Project Paths
    # -----------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    APP_DIR = PROJECT_ROOT / "app"

    DATASET_DIR = PROJECT_ROOT / "datasets"

    MODEL_DIR = PROJECT_ROOT / "trained_models"

    LOG_DIR = PROJECT_ROOT / "logs"

    # -----------------------------
    # Device
    # -----------------------------

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -----------------------------
    # Audio
    # -----------------------------

    SAMPLE_RATE = 16000

    N_MELS = 128

    TARGET_LENGTH = 100

    N_FFT = 2048

    HOP_LENGTH = 512

    # -----------------------------
    # Prediction
    # -----------------------------

    CONFIDENCE_THRESHOLD = 0.70


settings = Settings()