"""Exceptions subpackage."""

from app.audio.exceptions.audio_exceptions import (
    AudioError,
    AudioFormatError,
    AudioLoadError,
    AudioProcessingError,
)
from app.audio.exceptions.dataset_exceptions import (
    CorruptAudioFileError,
    DatasetError,
    DatasetNotFoundError,
    ProtocolParsingError,
)
from app.audio.exceptions.model_exceptions import (
    ComponentNotFoundError,
    DuplicateRegistrationError,
    ModelCheckpointError,
    ModelError,
    ModelNotFoundError,
    RegistryError,
)

__all__ = [
    "AudioError",
    "AudioLoadError",
    "AudioFormatError",
    "AudioProcessingError",
    "DatasetError",
    "DatasetNotFoundError",
    "ProtocolParsingError",
    "CorruptAudioFileError",
    "ModelError",
    "ModelNotFoundError",
    "ModelCheckpointError",
    "RegistryError",
    "DuplicateRegistrationError",
    "ComponentNotFoundError",
]
