"""
Centralized configuration for the DeepFake Detector.

Design decisions
────────────────
• Dataclass-based sub-configs for type safety and IDE support.
• ``device`` is a lazy property — torch is only imported when first accessed,
  avoiding CUDA initialisation overhead on every import.
• Environment variable overrides for CI / production flexibility.
• A module-level ``settings`` singleton is kept for backward compatibility.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Sub-Configurations
# ──────────────────────────────────────────────


@dataclass
class AudioConfig:
    """Audio processing hyper-parameters."""

    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    n_mels: int = int(os.getenv("N_MELS", "128"))
    n_fft: int = int(os.getenv("N_FFT", "2048"))
    hop_length: int = int(os.getenv("HOP_LENGTH", "512"))
    target_length: int = int(os.getenv("TARGET_LENGTH", "100"))


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    grad_accum_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.0001"))
    epochs: int = int(os.getenv("EPOCHS", "20"))
    num_workers: int = int(
        os.getenv("NUM_WORKERS", "0" if sys.platform == "win32" else "4")
    )
    early_stopping_patience: int = int(os.getenv("EARLY_STOPPING_PATIENCE", "5"))
    gradient_clip_norm: float = float(os.getenv("GRADIENT_CLIP_NORM", "1.0"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    seed: int = int(os.getenv("SEED", "42"))
    use_mixed_precision: bool = os.getenv("USE_MIXED_PRECISION", "true").lower() == "true"
    scheduler_t0: int = int(os.getenv("SCHEDULER_T0", "5"))


@dataclass
class ModelConfig:
    """Model architecture and inference parameters."""

    num_classes: int = 2
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    model_name: str = "aasist"
    video_model_name: str = "efficientnet_b4"
    model_version: str = "2.0"


@dataclass
class InferenceConfig:
    """Inference-specific parameters."""

    confidence_threshold_fake: float = float(
        os.getenv("THRESHOLD_FAKE", "0.7")
    )
    confidence_threshold_real: float = float(
        os.getenv("THRESHOLD_REAL", "0.3")
    )
    audio_fusion_weight: float = 0.60
    video_fusion_weight: float = 0.40


# ──────────────────────────────────────────────
# Main Settings
# ──────────────────────────────────────────────


@dataclass
class Settings:
    """
    Top-level project configuration.

    Access sub-configs via ``settings.audio``, ``settings.training``, etc.
    """

    # ── Sub-configs ───────────────────────────
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # ── Paths ─────────────────────────────────
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # ── Fusion Weights (shortcuts) ────────────
    AUDIO_WEIGHT: float = 0.60
    VIDEO_WEIGHT: float = 0.40

    def __post_init__(self) -> None:
        """Derive dependent paths and create required directories."""
        # Dataset root paths
        self.DATASET_DIR: Path = self.project_root / "datasets"
        self.AUDIO_DATASET_DIR: Path = self.DATASET_DIR / "audio"
        self.VIDEO_DATASET_DIR: Path = self.DATASET_DIR / "video"

        # Audio datasets (ASVspoof2019 & ASVspoof2021)
        self.ASVSPOOF2019_DIR: Path = self.AUDIO_DATASET_DIR / "asvspoof2019"
        self.ASVSPOOF2021_DIR: Path = self.AUDIO_DATASET_DIR / "asvspoof2021"

        self.TRAIN_AUDIO_DIR: Path = self.ASVSPOOF2019_DIR / "ASVspoof2019_LA_train" / "flac"
        self.VAL_AUDIO_DIR: Path = self.ASVSPOOF2019_DIR / "ASVspoof2019_LA_dev" / "flac"
        self.TEST_AUDIO_DIR: Path = self.ASVSPOOF2019_DIR / "ASVspoof2019_LA_eval" / "flac"

        protocols = self.ASVSPOOF2019_DIR / "ASVspoof2019_LA_cm_protocols"
        self.TRAIN_PROTOCOL_FILE: Path = protocols / "ASVspoof2019.LA.cm.train.trn.txt"
        self.VAL_PROTOCOL_FILE: Path = protocols / "ASVspoof2019.LA.cm.dev.trl.txt"
        self.TEST_PROTOCOL_FILE: Path = protocols / "ASVspoof2019.LA.cm.eval.trl.txt"

        # Video datasets
        self.FACEFORENSICS_DIR: Path = self.VIDEO_DATASET_DIR / "faceforensics"
        self.CELEB_DFV2_DIR: Path = self.VIDEO_DATASET_DIR / "celebdfv2"
        self.VIDEO_CACHE_DIR: Path = self.VIDEO_DATASET_DIR / "cache"
        self.VIDEO_PROCESSED_DIR: Path = self.VIDEO_DATASET_DIR / "processed"

        # Model directory
        self.MODEL_DIR: Path = self.project_root / "trained_models"
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_SAVE_PATH: Path = self.MODEL_DIR / "best_model.pth"

        # Log directory
        self.LOG_DIR: Path = self.project_root / "logs"
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Cache directory for pre-computed features
        self.CACHE_DIR: Path = self.project_root / "cache"

        # TensorBoard directory
        self.TENSORBOARD_DIR: Path = self.project_root / "runs"

        # ── Backward-compat aliases ───────────
        self.SAMPLE_RATE: int = self.audio.sample_rate
        self.N_MELS: int = self.audio.n_mels
        self.TIME_FRAMES: int = self.audio.target_length
        self.BATCH_SIZE: int = self.training.batch_size
        self.LEARNING_RATE: float = self.training.learning_rate
        self.EPOCHS: int = self.training.epochs

    # ── Lazy device ───────────────────────────

    _device: Optional[object] = field(default=None, init=False, repr=False)

    @property
    def DEVICE(self) -> "torch.device":  # type: ignore[name-defined]
        """Return the compute device, lazily importing torch."""
        if self._device is None:
            import torch

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._device  # type: ignore[return-value]

    # ── Validation ────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of configuration warnings (empty == all good)."""
        warnings: list[str] = []
        if self.training.batch_size < 1:
            warnings.append("batch_size must be >= 1")
        if self.training.learning_rate <= 0:
            warnings.append("learning_rate must be > 0")
        if self.audio.sample_rate < 8000:
            warnings.append("sample_rate below 8000 Hz is unusual for speech")
        if self.training.num_workers < 0:
            warnings.append("num_workers must be >= 0")
        if abs(self.AUDIO_WEIGHT + self.VIDEO_WEIGHT - 1.0) > 1e-6:
            warnings.append("AUDIO_WEIGHT + VIDEO_WEIGHT should sum to 1.0")
        return warnings


# ──────────────────────────────────────────────
# Module-level singleton (backward compatible)
# ──────────────────────────────────────────────

settings = Settings()