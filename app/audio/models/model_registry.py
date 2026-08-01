"""Backward-compatibility shim for model_registry."""

from app.audio.registry.model_registry import ModelRegistry, model_registry

__all__ = [
    "ModelRegistry",
    "model_registry",
]
