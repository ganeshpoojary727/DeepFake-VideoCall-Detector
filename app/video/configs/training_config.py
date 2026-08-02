"""Video training configuration dataclass module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from app.video.constants.video_constants import (
    DEFAULT_DEVICE,
    VIDEO_LOGS_DIR,
    VIDEO_MODELS_DIR,
)
from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class VideoTrainingConfig:
    """Configuration settings for video model training pipeline execution."""

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    loss_name: str = "cross_entropy"
    device: str = DEFAULT_DEVICE
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 1.0
    gradient_clip_norm: float = 1.0
    clip_duration_seconds: float = 20.0
    frames_sampled: int = 16
    sampling_strategy: str = "uniform"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    checkpoint_dir: Path = field(default_factory=lambda: Path(VIDEO_MODELS_DIR))
    log_dir: Path = field(default_factory=lambda: Path(VIDEO_LOGS_DIR))
    tensorboard_dir: Path = field(default_factory=lambda: Path(VIDEO_LOGS_DIR) / "tensorboard")
    save_top_k: int = 3
    save_best_metric: str = "val_loss"
    num_workers: int = 4
    seed: int = 42
    log_interval: int = 10
    label_smoothing: float = 0.1
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve directory paths."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.tensorboard_dir = Path(self.tensorboard_dir)

    def validate(self) -> None:
        """Validate hyperparameter ranges."""
        if self.epochs <= 0:
            raise ConfigurationError(f"Epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ConfigurationError(f"Learning rate must be positive, got {self.learning_rate}")
        if self.num_workers < 0:
            raise ConfigurationError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )


# Training config alias
TrainingConfig = VideoTrainingConfig
