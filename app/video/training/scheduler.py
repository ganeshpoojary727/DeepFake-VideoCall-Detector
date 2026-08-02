"""Scheduler module wrapper for video model learning rate scheduling."""

from __future__ import annotations

from typing import Any
import torch

from app.video.configs.training_config import VideoTrainingConfig
from app.video.training.scheduler_factory import SchedulerFactory


def create_scheduler(optimizer: torch.optim.Optimizer, config: VideoTrainingConfig) -> Any:
    """Create learning rate scheduler from optimizer and config.

    Args:
        optimizer: PyTorch optimizer instance.
        config: Video training config.

    Returns:
        Any: Instantiated learning rate scheduler.
    """
    return SchedulerFactory.create_scheduler(optimizer=optimizer, config=config)
