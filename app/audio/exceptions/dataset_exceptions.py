"""Dataset domain custom exception classes.

Defines custom exceptions for dataset loading, protocol parsing, missing data,
and corrupt sample errors.
"""

from __future__ import annotations

from app.audio.exceptions.audio_exceptions import AudioError


class DatasetError(AudioError):
    """Base exception for all dataset operations."""

    pass


class DatasetNotFoundError(DatasetError):
    """Exception raised when a dataset directory or required dataset split is missing."""

    pass


class ProtocolParsingError(DatasetError):
    """Exception raised when parsing dataset protocol/annotation text files fails."""

    pass


class CorruptAudioFileError(DatasetError):
    """Exception raised when an audio file in a dataset is corrupt or unreadable."""

    pass
