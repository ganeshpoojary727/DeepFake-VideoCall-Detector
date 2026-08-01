"""
Audio subsystem logger utility wrapper.

Provides the AudioLogger class for producing standardized, formatted logger instances
across audio preprocessing, model training, feature extraction, and evaluation.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.utils.logger import get_logger


class AudioLogger:
    """
    Factory and utility wrapper for audio subsystem logging.

    Parameters
    ----------
    name : str
        Name of the module requesting logging capabilities.
    """

    def __init__(self, name: str = "app.audio") -> None:
        self._logger = get_logger(name)

    @property
    def logger(self) -> logging.Logger:
        """Access underlying logging.Logger instance."""
        return self._logger

    @staticmethod
    def get(name: str) -> logging.Logger:
        """
        Static helper to retrieve a named Logger instance.

        Parameters
        ----------
        name : str
            Module or component name.

        Returns
        -------
        logging.Logger
            Configured Logger instance.
        """
        return get_logger(f"app.audio.{name}" if not name.startswith("app.audio") else name)

    def info(self, msg: str, *args: str) -> None:
        """Log info level message."""
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: str) -> None:
        """Log warning level message."""
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: str) -> None:
        """Log error level message."""
        self._logger.error(msg, *args)

    def debug(self, msg: str, *args: str) -> None:
        """Log debug level message."""
        self._logger.debug(msg, *args)
