"""Video subsystem logging utility module."""

from __future__ import annotations

import logging
import sys
from typing import Optional


class VideoLogger:
    """Logger wrapper for video processing and model training events."""

    _logger: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "VideoAI") -> logging.Logger:
        """Get or configure logger instance.

        Args:
            name: Logger name identifier.

        Returns:
            logging.Logger: Configured logger.
        """
        if cls._logger is None:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            cls._logger = logger
        return cls._logger


def get_video_logger(name: str = "VideoAI") -> logging.Logger:
    """Helper function to retrieve configured video logger instance."""
    return VideoLogger.get_logger(name)
