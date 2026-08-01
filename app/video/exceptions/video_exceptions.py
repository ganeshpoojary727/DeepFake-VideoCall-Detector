"""Video AI subsystem exception hierarchy.

Defines all exception types used across video preprocessing, datasets,
models, training, evaluation, registries, and pipelines.
"""

from __future__ import annotations


class VideoException(Exception):
    """Base exception class for all Video AI subsystem errors."""

    pass


class ConfigurationError(VideoException):
    """Raised when configuration validation or initialization fails."""

    pass


class DatasetError(VideoException):
    """Raised when video dataset loading, metadata parsing, or sampling fails."""

    pass


class PreprocessingError(VideoException):
    """Raised when frame extraction, face cropping, or tensor conversion fails."""

    pass


class AugmentationError(VideoException):
    """Raised when applying spatial or temporal video augmentations fails."""

    pass


class ModelError(VideoException):
    """Raised when video model instantiation, forward pass, or loading fails."""

    pass


class TrainingError(VideoException):
    """Raised when video model training step, loss calculation, or optimization fails."""

    pass


class PipelineError(VideoException):
    """Raised when video training, validation, or inference pipeline fails."""

    pass


class RegistryError(VideoException):
    """Base exception for component registry failures."""

    pass


class ComponentNotFoundError(RegistryError):
    """Raised when requesting an unregistered component key from a registry."""

    pass


class DuplicateRegistrationError(RegistryError):
    """Raised when registering an existing component key without overwrite=True."""

    pass
