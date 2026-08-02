"""Model and registry custom exception classes for video AI subsystem."""

from __future__ import annotations

from app.video.exceptions.video_exceptions import VideoError


class ModelError(VideoError):
    """Base exception for all video model-related failures."""

    pass


class ModelNotFoundError(ModelError):
    """Exception raised when a requested video model architecture is not found."""

    pass


class ModelInitializationError(ModelError):
    """Exception raised when initializing or configuring a video model fails."""

    pass


class ModelCheckpointError(ModelError):
    """Exception raised when saving or loading a video model checkpoint fails."""

    pass


# Alias for prompt requirement
CheckpointError = ModelCheckpointError


class RegistryError(VideoError):
    """Base exception for video component registry failures."""

    pass


class DuplicateRegistrationError(RegistryError):
    """Exception raised when registering a component with a duplicate identifier."""

    pass


class ComponentNotFoundError(RegistryError):
    """Exception raised when retrieving an unregistered component from a registry."""

    pass
