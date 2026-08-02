"""Video AI subsystem exception hierarchy.

Defines all exception types used across video preprocessing, datasets,
models, training, evaluation, registries, and pipelines.
"""

from __future__ import annotations


class VideoError(Exception):
    """Base exception class for all Video AI subsystem errors."""

    pass


# Alias for VideoError
VideoException = VideoError


class ConfigurationError(VideoError):
    """Raised when configuration validation or initialization fails."""

    pass


class DatasetError(VideoError):
    """Raised when video dataset loading, metadata parsing, or sampling fails."""

    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a video dataset directory or split is missing."""

    pass


# Alias for prompt requirement
DatasetNotFound = DatasetNotFoundError


class ProtocolParsingError(DatasetError):
    """Raised when parsing dataset protocol/annotation metadata fails."""

    pass


class CorruptVideoFileError(DatasetError):
    """Raised when a video file in a dataset is corrupt or unreadable."""

    pass


class PreprocessingError(VideoError):
    """Raised when frame extraction, face cropping, or tensor conversion fails."""

    pass


class FrameExtractionError(PreprocessingError):
    """Raised when frame extraction from a video file or stream fails."""

    pass


class FaceDetectionError(PreprocessingError):
    """Raised when face detection in a video frame fails."""

    pass


class FaceAlignmentError(PreprocessingError):
    """Raised when facial landmark alignment fails."""

    pass


class AugmentationError(VideoError):
    """Raised when applying spatial or temporal video augmentations fails."""

    pass


class ModelError(VideoError):
    """Raised when video model instantiation, forward pass, or loading fails."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested video model architecture is not found."""

    pass


class ModelInitializationError(ModelError):
    """Raised when initializing or configuring a video model fails."""

    pass


class ModelCheckpointError(ModelError):
    """Raised when saving or loading a video model checkpoint fails."""

    pass


# Alias for prompt requirement
CheckpointError = ModelCheckpointError


class TrainingError(VideoError):
    """Raised when video model training step, loss calculation, or optimization fails."""

    pass


class PipelineError(VideoError):
    """Raised when video training, validation, or inference pipeline fails."""

    pass


class RegistryError(VideoError):
    """Base exception for component registry failures."""

    pass


class ComponentNotFoundError(RegistryError):
    """Raised when requesting an unregistered component key from a registry."""

    pass


class DuplicateRegistrationError(RegistryError):
    """Raised when registering an existing component key without overwrite=True."""

    pass
