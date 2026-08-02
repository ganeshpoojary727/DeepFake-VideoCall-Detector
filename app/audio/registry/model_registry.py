"""Model architecture registry module.

Provides ModelRegistry for registering, looking up, and instantiating audio
neural network models (AASIST, RawNet, ECAPA, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Type
import torch.nn as nn

from app.audio.exceptions.model_exceptions import ComponentNotFoundError
from app.audio.models.aasist import AASIST
from app.audio.registry.base_registry import BaseRegistry


class ModelRegistry(BaseRegistry[nn.Module]):
    """Registry for neural network model architectures."""

    _registry: Dict[str, Type[nn.Module]] = {}

    def __init__(self) -> None:
        super().__init__(name="ModelRegistry")
        self.register("AASIST", AASIST, overwrite=True)
        self.register("aasist", AASIST, overwrite=True)

    @classmethod
    def list_models(cls) -> List[str]:
        """Class method compatibility alias for listing registered model names."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Class method compatibility alias for instantiating registered models."""
        for k, model_cls in cls._registry.items():
            if k.lower() == name.lower():
                return model_cls(**kwargs)
        raise KeyError(f"Model '{name}' not found in ModelRegistry.")

    def register(
        self,
        name: str,
        component: Type[nn.Module],
        overwrite: bool = False,
    ) -> None:
        """Register component instance and update class registry dict."""
        super().register(name, component, overwrite=overwrite)
        self._registry[name] = component
        self._registry[name.lower()] = component


# Initialize class-level defaults
ModelRegistry._registry["AASIST"] = AASIST
ModelRegistry._registry["aasist"] = AASIST

# Default global instance for model architectures
model_registry = ModelRegistry()
