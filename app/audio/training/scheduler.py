"""
Learning rate scheduler factory module.

Provides the SchedulerFactory class for configuring PyTorch learning rate schedulers
(CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, StepLR).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("training.scheduler")


class SchedulerFactory:
    """
    Factory for instantiating PyTorch learning rate schedulers.

    Parameters
    ----------
    config : AudioTrainingConfig
        Training configuration specifying scheduler parameters and total epochs.
    """

    def __init__(self, config: AudioTrainingConfig) -> None:
        self.config = config

    def create_scheduler(
        self,
        optimizer: Optimizer,
        steps_per_epoch: Optional[int] = None,
    ) -> Any:
        """
        Create PyTorch learning rate scheduler for specified optimizer.

        Parameters
        ----------
        optimizer : Optimizer
            Target PyTorch optimizer.
        steps_per_epoch : Optional[int]
            Number of iterations per epoch (required for OneCycleLR).

        Returns
        -------
        Any
            PyTorch LRScheduler or ReduceLROnPlateau instance.

        Raises
        ------
        ValueError
            If an unsupported scheduler name or missing required parameter is provided.
        """
        name = self.config.scheduler_name.lower().strip()
        logger.info("Creating LR scheduler '%s'", name)

        if name in ("cosine", "cosineannealing"):
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6,
            )
        if name in ("plateau", "reducelronplateau"):
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
        if name in ("step", "steplr"):
            return StepLR(
                optimizer,
                step_size=5,
                gamma=0.5,
            )
        if name in ("onecycle", "onecyclelr"):
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required for OneCycleLR")
            return OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
            )

        raise ValueError(
            f"Unsupported scheduler type: '{name}'. Supported: cosine, plateau, step, onecycle."
        )
