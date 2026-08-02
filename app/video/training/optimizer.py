"""Optimizer module wrapper for video model training."""

from __future__ import annotations

from typing import Any, Dict
import torch
import torch.nn as nn

from app.video.configs.training_config import VideoTrainingConfig
from app.video.training.optimizer_factory import OptimizerFactory


def create_optimizer(model: nn.Module, config: VideoTrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer from model and training config.

    Args:
        model: Target PyTorch model.
        config: Video training config.

    Returns:
        torch.optim.Optimizer: Instantiated PyTorch optimizer.
    """
    return OptimizerFactory.create_optimizer(model=model, config=config)
