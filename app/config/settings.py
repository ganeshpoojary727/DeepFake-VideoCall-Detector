"""
Application configuration.

All project-wide settings should be defined here.
"""

from pathlib import Path
import torch


class Settings:

    def __init__(self):

        # ------------------------
        # Project Directories
        # ------------------------

        self.PROJECT_ROOT = Path(__file__).resolve().parents[2]

        self.APP_DIR = self.PROJECT_ROOT / "app"

        self.DATASET_DIR = self.PROJECT_ROOT / "datasets"

        self.MODEL_DIR = self.PROJECT_ROOT / "trained_models"

        self.LOG_DIR = self.PROJECT_ROOT / "logs"

        # ------------------------
        # Device
        # ------------------------

        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------
        # Audio Configuration
        # ------------------------

        self.SAMPLE_RATE = 16000

        self.N_MELS = 128

        self.N_FFT = 2048

        self.HOP_LENGTH = 512

        self.TARGET_LENGTH = 100

        # ------------------------
        # Prediction
        # ------------------------

        self.CONFIDENCE_THRESHOLD = 0.70


settings = Settings()