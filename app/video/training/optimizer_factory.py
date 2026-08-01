"""PyTorch optimizer factory module."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Type
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from app.video.exceptions.video_exceptions import ConfigurationError
from app.video.registry.video_registries import optimizer_registry


class OptimizerFactory:
    """Factory for creating registered PyTorch optimizer instances."""

    _mapping: Dict[str, Type[Optimizer]] = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
    }

    @classmethod
    def create(
        cls,
        name: str,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ) -> Optimizer:
        """Create PyTorch optimizer by name lookup.

        Args:
            name: Optimizer key name ("adam", "adamw", "sgd").
            params: Model parameter iterator.
            lr: Learning rate.
            weight_decay: Weight decay penalty.

        Returns:
            Optimizer: Instantiated PyTorch optimizer.
        """
        key = name.lower().strip()
        if key in cls._mapping:
            opt_cls = cls._mapping[key]
        else:
            try:
                opt_cls = optimizer_registry.get(key)
            except Exception as err:
                raise ConfigurationError(f"Unsupported optimizer name '{name}'") from err

        return opt_cls(params, lr=lr, weight_decay=weight_decay, **kwargs)


# Register defaults in global registry
for opt_key, opt_class in OptimizerFactory._mapping.items():
    optimizer_registry.register(opt_key, opt_class, overwrite=True)
