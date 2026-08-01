"""
Model registry — factory pattern for model creation and versioning.

Usage
-----
    from app.audio.models.model_registry import ModelRegistry

    model = ModelRegistry.create("LightCNN")
    model = ModelRegistry.create("DeepFakeCNN", num_classes=2)
    names = ModelRegistry.list_models()
"""

from __future__ import annotations

from typing import Dict, List, Type

import torch.nn as nn

from app.audio.models.cnn_model import DeepFakeCNN, LightCNN
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Factory for registering and instantiating model architectures.

    New models can be added without modifying existing code:

    .. code-block:: python

        @ModelRegistry.register("MyModel")
        class MyModel(nn.Module): ...
    """

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class under *name*."""
        def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            cls._registry[name] = model_cls
            logger.debug("Registered model: %s", name)
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """
        Instantiate a registered model by name.

        Parameters
        ----------
        name : str
            Model name as registered in the registry.
        **kwargs
            Arguments forwarded to the model constructor.

        Returns
        -------
        nn.Module
            A new model instance.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Unknown model '{name}'. Available: {available}"
            )
        model = cls._registry[name](**kwargs)
        logger.info("Created model: %s (%s params)", name, f"{sum(p.numel() for p in model.parameters()):,}")
        return model

    @classmethod
    def list_models(cls) -> List[str]:
        """Return a sorted list of registered model names."""
        return sorted(cls._registry.keys())


# ──────────────────────────────────────────────
# Register built-in models
# ──────────────────────────────────────────────

ModelRegistry._registry["DeepFakeCNN"] = DeepFakeCNN
ModelRegistry._registry["LightCNN"] = LightCNN
