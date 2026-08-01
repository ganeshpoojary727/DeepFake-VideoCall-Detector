"""Audio domain custom exception classes.

Defines the base AudioError hierarchy and specific exceptions for audio loading,
format validation, and signal processing failures.
"""

from __future__ import annotations


class AudioError(Exception):
    """Base exception class for all audio subsystem errors."""

    pass


class AudioLoadError(AudioError):
    """Exception raised when an audio file fails to load from disk."""

    pass


class AudioFormatError(AudioError):
    """Exception raised when an audio file has an unsupported extension or format."""

    pass


class AudioProcessingError(AudioError):
    """Exception raised when audio signal processing operations fail."""

    pass
