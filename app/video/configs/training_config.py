"""Video training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.video.exceptions.video_exceptions import ConfigurationError


@dataclass
class VideoTrainingConfig:
    """Configuration settings for video model training execution."""

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    loss_name: str = "cross_entropy"
    device: str = "cuda"
    use_amp: bool = True
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    checkpoint_dir: str = "checkpoints/video"
    save_top_k: int = 3
    num_workers: int = 4
    seed: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration hyperparameter ranges."""
        if self.epochs <= 0:
            raise ConfigurationError(f"Epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ConfigurationError(f"Learning rate must be positive, got {self.learning_rate}")
