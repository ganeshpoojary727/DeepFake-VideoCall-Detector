"""Config-driven model builder module.

Provides ModelBuilder for constructing neural network models from AudioModelConfig
without hardcoding model-specific architecture parameters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch.nn as nn

from app.audio.configs.model_config import AudioModelConfig
from app.audio.registry.model_registry import ModelRegistry, model_registry
from app.audio.utils.logger import AudioLogger

logger = AudioLogger.get("builders.model")


class ModelBuilder:
    """Builder for instantiating neural network models from configuration objects.

    Args:
        registry (Optional[ModelRegistry]): Custom model registry instance.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self._registry = registry or model_registry

    @property
    def registry(self) -> ModelRegistry:
        """Get attached model registry instance."""
        return self._registry

    def build(
        self,
        config: AudioModelConfig,
        override_params: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Construct and return a neural network model from config.

        Args:
            config (AudioModelConfig): Model architecture configuration object.
            override_params (Optional[Dict[str, Any]]): Optional hyperparameter overrides.

        Returns:
            nn.Module: Instantiated PyTorch neural network model.
        """
        model_cls = self._registry.get(config.model_name)
        kwargs: Dict[str, Any] = {
            "num_classes": config.num_classes,
        }
        if override_params:
            kwargs.update(override_params)

        logger.info("Building model '%s' with params: %s", config.model_name, kwargs)
        return model_cls(**kwargs)
