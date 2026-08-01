"""Config-driven trainer infrastructure builder module.

Provides TrainerBuilder for constructing models, optimizers, learning rate schedulers,
and loss functions from configuration dataclasses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from app.audio.builders.model_builder import ModelBuilder
from app.audio.configs.model_config import AudioModelConfig
from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.registry.loss_registry import LossRegistry, loss_registry
from app.audio.registry.optimizer_registry import OptimizerRegistry, optimizer_registry
from app.audio.registry.scheduler_registry import SchedulerRegistry, scheduler_registry
from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("builders.trainer")


class TrainerBuilder:
    """Builder for assembling model training components from dataclass configurations."""

    def __init__(
        self,
        model_builder: Optional[ModelBuilder] = None,
        opt_registry: Optional[OptimizerRegistry] = None,
        sched_registry: Optional[SchedulerRegistry] = None,
        l_registry: Optional[LossRegistry] = None,
    ) -> None:
        self.model_builder = model_builder or ModelBuilder()
        self.opt_registry = opt_registry or optimizer_registry
        self.sched_registry = sched_registry or scheduler_registry
        self.loss_registry = l_registry or loss_registry

    def build_optimizer(
        self,
        model: nn.Module,
        config: AudioTrainingConfig,
    ) -> Optimizer:
        """Instantiate optimizer for model parameters based on training config.

        Args:
            model (nn.Module): Target PyTorch model.
            config (AudioTrainingConfig): Training configuration dataclass.

        Returns:
            Optimizer: Initialized optimizer instance.
        """
        opt_cls = self.opt_registry.get(config.optimizer_name)
        logger.info(
            "Building optimizer '%s' with lr=%.5f, weight_decay=%.6f",
            config.optimizer_name,
            config.learning_rate,
            config.weight_decay,
        )
        return opt_cls(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def build_scheduler(
        self,
        optimizer: Optimizer,
        config: AudioTrainingConfig,
    ) -> Any:
        """Instantiate learning rate scheduler for optimizer based on training config.

        Args:
            optimizer (Optimizer): Target optimizer instance.
            config (AudioTrainingConfig): Training configuration dataclass.

        Returns:
            Any: Initialized learning rate scheduler instance.
        """
        sched_cls = self.sched_registry.get(config.scheduler_name)
        logger.info("Building scheduler '%s'", config.scheduler_name)
        if config.scheduler_name.lower().strip() in ("cosine", "cosineannealing"):
            return sched_cls(optimizer, T_max=config.epochs)
        if config.scheduler_name.lower().strip() in ("plateau", "reducelronplateau"):
            return sched_cls(optimizer, mode="min", factor=0.5, patience=3)
        return sched_cls(optimizer)

    def build_loss_function(
        self,
        loss_name: str = "cross_entropy",
        weight: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Instantiate loss criterion based on registry lookup.

        Args:
            loss_name (str): Identifier for loss function.
            weight (Optional[torch.Tensor]): Class weights tensor for loss balancing.

        Returns:
            nn.Module: Initialized PyTorch loss module.
        """
        loss_cls = self.loss_registry.get(loss_name)
        if weight is not None and hasattr(loss_cls, "__init__"):
            try:
                return loss_cls(weight=weight)
            except TypeError:
                pass
        return loss_cls()
