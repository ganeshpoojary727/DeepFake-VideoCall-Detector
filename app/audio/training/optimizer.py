"""
Optimizer construction factory module.

Provides the OptimizerFactory class for instantiating PyTorch optimizer objects
(Adam, AdamW, SGD) based on AudioTrainingConfig hyper-parameters.
"""

from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from app.audio.configs.training_config import AudioTrainingConfig
from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("training.optimizer")


class OptimizerFactory:
    """
    Factory for instantiating PyTorch optimizers.

    Parameters
    ----------
    config : AudioTrainingConfig
        Training configuration specifying learning rate, weight decay, and optimizer name.
    """

    def __init__(self, config: AudioTrainingConfig) -> None:
        self.config = config

    def create_optimizer(
        self,
        model_or_params: Union[nn.Module, Iterable[torch.nn.Parameter]],
    ) -> Optimizer:
        """
        Create PyTorch optimizer instance for model parameters.

        Parameters
        ----------
        model_or_params : Union[nn.Module, Iterable[torch.nn.Parameter]]
            PyTorch nn.Module or iterable of parameter Tensors.

        Returns
        -------
        Optimizer
            Configured PyTorch optimizer instance.

        Raises
        ------
        ValueError
            If an unsupported optimizer name is configured.
        """
        params = (
            model_or_params.parameters()
            if isinstance(model_or_params, nn.Module)
            else model_or_params
        )

        name = self.config.optimizer_name.lower().strip()
        logger.info(
            "Creating optimizer '%s' (lr=%.5f, weight_decay=%.6f)",
            name,
            self.config.learning_rate,
            self.config.weight_decay,
        )

        if name == "adam":
            return Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if name == "adamw":
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if name == "sgd":
            return SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )

        raise ValueError(f"Unsupported optimizer type: '{name}'. Supported: adam, adamw, sgd.")
