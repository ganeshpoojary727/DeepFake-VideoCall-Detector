"""Audio model training configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.audio.constants.audio_constants import (
    AUDIO_LOGS_DIR,
    AUDIO_MODELS_DIR,
    PRODUCTION_AUDIO_MODEL,
)


@dataclass
class AudioTrainingConfig:
    """Configuration parameters for audio AASIST model training pipeline."""

    model_name: str = PRODUCTION_AUDIO_MODEL
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 50
    weight_decay: float = 1e-4
    use_amp: bool = True
    grad_accum_steps: int = 4
    use_checkpointing: bool = True
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    seed: int = 42
    optimizer_name: str = "adamw"  # 'adamw', 'lion', 'sgd'
    scheduler_name: str = "cosine"  # 'cosine', 'onecycle', 'warmup_cosine', 'plateau'
    loss_name: str = "cross_entropy"  # 'cross_entropy', 'focal', 'label_smoothing', 'class_balanced'
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.999
    checkpoint_dir: Path = field(default_factory=lambda: Path(AUDIO_MODELS_DIR))
    log_dir: Path = field(default_factory=lambda: Path(AUDIO_LOGS_DIR))
    tensorboard_dir: Path = field(default_factory=lambda: Path(AUDIO_LOGS_DIR) / "tensorboard")
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Resolve paths and validate parameters."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.tensorboard_dir = Path(self.tensorboard_dir)
        self.validate()

    def validate(self) -> None:
        """Validate training hyperparameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive, got {self.grad_accum_steps}")
