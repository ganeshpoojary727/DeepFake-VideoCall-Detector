"""Optimizer construction factory module for audio models."""

from __future__ import annotations

from typing import Any, Iterable, Union
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.registry.optimizer_registry import optimizer_registry
from app.audio.training.optimizers.lion_optimizer import Lion


class OptimizerFactory:
    """Factory for instantiating PyTorch optimizers (AdamW, Lion, SGD)."""

    def __init__(self, config: AudioTrainingConfig) -> None:
        self.config = config

    def create_optimizer(
        self,
        model_or_params: Union[nn.Module, Iterable[torch.nn.Parameter]],
    ) -> Optimizer:
        """Create PyTorch optimizer instance for model parameters."""
        params = (
            model_or_params.parameters()
            if isinstance(model_or_params, nn.Module)
            else model_or_params
        )

        name = self.config.optimizer_name.lower().strip()

        if name == "adamw":
            return AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif name == "lion":
            return Lion(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif name == "sgd":
            return SGD(params, lr=self.config.learning_rate, momentum=0.9, weight_decay=self.config.weight_decay)
        elif name == "adam":
            return Adam(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            try:
                opt_cls = optimizer_registry.get(name)
                return opt_cls(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            except Exception as err:
                raise ValueError(f"Unsupported optimizer type: '{name}'. Supported: adamw, lion, sgd, adam.") from err


# Register default optimizers in global optimizer_registry
optimizer_registry.register("adamw", AdamW, overwrite=True)
optimizer_registry.register("lion", Lion, overwrite=True)
optimizer_registry.register("sgd", SGD, overwrite=True)
