"""Model and registry custom exception classes.

Defines custom exceptions for neural network model creation, loading, checkpointing,
and component registration errors.
"""

from __future__ import annotations

from app.audio.exceptions.audio_exceptions import AudioError


class ModelError(AudioError):
    """Base exception for all model-related failures."""

    pass


class ModelNotFoundError(ModelError):
    """Exception raised when a requested model architecture is not found."""

    pass


class ModelCheckpointError(ModelError):
    """Exception raised when saving or loading a model checkpoint fails."""

    pass


class RegistryError(AudioError):
    """Base exception for component registry failures."""

    pass


class DuplicateRegistrationError(RegistryError):
    """Exception raised when registering a component with a duplicate identifier."""

    pass


class ComponentNotFoundError(RegistryError):
    """Exception raised when retrieving an unregistered component from a registry."""

    pass
