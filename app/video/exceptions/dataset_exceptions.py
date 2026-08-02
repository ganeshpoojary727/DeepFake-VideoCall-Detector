"""Dataset domain custom exception classes for video AI subsystem."""

from __future__ import annotations

from app.video.exceptions.video_exceptions import VideoError


class DatasetError(VideoError):
    """Base exception for all video dataset operations."""

    pass


class DatasetNotFoundError(DatasetError):
    """Exception raised when a video dataset directory or split is missing."""

    pass


# Alias for prompt requirement
DatasetNotFound = DatasetNotFoundError


class ProtocolParsingError(DatasetError):
    """Exception raised when parsing dataset protocol/annotation metadata fails."""

    pass


class CorruptVideoFileError(DatasetError):
    """Exception raised when a video file in a dataset is corrupt or unreadable."""

    pass
