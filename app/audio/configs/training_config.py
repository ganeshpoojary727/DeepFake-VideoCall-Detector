"""
Audio model training configuration definitions.

Provides structured dataclass configurations for model training, optimization,
learning rate scheduling, and checkpoint retention settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config.settings import settings


@dataclass
class AudioTrainingConfig:
    """
    Configuration parameters for audio model training pipeline.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for training and validation.
    learning_rate : float
        Initial learning rate for the optimizer.
    epochs : int
        Total training epochs.
    weight_decay : float
        L2 regularization weight decay factor.
    use_amp : bool
        Whether to enable Automatic Mixed Precision (AMP).
    gradient_clip_norm : float
        Maximum norm for gradient clipping.
    early_stopping_patience : int
        Epochs without validation improvement before stopping.
    seed : int
        Random seed for reproducibility.
    optimizer_name : str
        Optimizer identifier ('adam', 'adamw', 'sgd').
    scheduler_name : str
        Scheduler identifier ('cosine', 'plateau', 'step', 'onecycle').
    checkpoint_dir : Path
        Directory path for storing model checkpoints.
    log_interval : int
        Batch frequency for logging training progress.
    """

    batch_size: int = field(default_factory=lambda: settings.training.batch_size)
    learning_rate: float = field(default_factory=lambda: settings.training.learning_rate)
    epochs: int = field(default_factory=lambda: settings.training.epochs)
    weight_decay: float = field(default_factory=lambda: settings.training.weight_decay)
    use_amp: bool = field(default_factory=lambda: settings.training.use_mixed_precision)
    gradient_clip_norm: float = field(
        default_factory=lambda: settings.training.gradient_clip_norm
    )
    early_stopping_patience: int = field(
        default_factory=lambda: settings.training.early_stopping_patience
    )
    seed: int = field(default_factory=lambda: settings.training.seed)
    optimizer_name: str = "adamw"
    scheduler_name: str = "cosine"
    checkpoint_dir: Path = field(default_factory=lambda: Path(settings.MODEL_SAVE_PATH).parent)
    log_interval: int = 10

    def __post_init__(self) -> None:
        """Validate configuration parameters and resolve paths."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.validate()

    def validate(self) -> None:
        """
        Validate training hyperparameters.

        Raises
        ------
        ValueError
            If any hyperparameter value is out of valid bounds.
        """
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}"
            )
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )
