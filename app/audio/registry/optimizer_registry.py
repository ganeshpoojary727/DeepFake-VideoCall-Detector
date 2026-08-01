"""Optimizer registry module.

Provides OptimizerRegistry for registering and looking up PyTorch optimizer
classes (Adam, AdamW, SGD, RMSprop, etc.).
"""

from __future__ import annotations

from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from app.audio.registry.base_registry import BaseRegistry


class OptimizerRegistry(BaseRegistry[Optimizer]):
    """Registry for PyTorch optimizer classes."""

    def __init__(self) -> None:
        super().__init__(name="OptimizerRegistry")
        self.register("adam", Adam)
        self.register("adamw", AdamW)
        self.register("sgd", SGD)
        self.register("rmsprop", RMSprop)


# Default global instance for optimizers
optimizer_registry = OptimizerRegistry()
