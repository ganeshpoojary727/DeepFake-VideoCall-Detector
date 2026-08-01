"""Audio subsystem logging utility module."""

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from app.audio.constants.audio_constants import AUDIO_LOGS_DIR


class AudioLogger:
    """Configures structured file and console loggers for audio training operations."""

    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "AudioAI", log_dir: str | Path = AUDIO_LOGS_DIR) -> logging.Logger:
        """Get or configure logger instance with ISO timestamps and file handlers."""
        if cls._instance is None:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            os.makedirs(log_dir, exist_ok=True)

            if not logger.handlers:
                # Console Handler
                console = logging.StreamHandler(sys.stdout)
                fmt = logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                console.setFormatter(fmt)
                logger.addHandler(console)

                # File Handlers
                train_file = logging.FileHandler(Path(log_dir) / "training.log", encoding="utf-8")
                train_file.setFormatter(fmt)
                logger.addHandler(train_file)

                err_file = logging.FileHandler(Path(log_dir) / "error.log", encoding="utf-8")
                err_file.setLevel(logging.ERROR)
                err_file.setFormatter(fmt)
                logger.addHandler(err_file)

            cls._instance = logger
        return cls._instance

    @classmethod
    def get(cls, name: str = "AudioAI") -> logging.Logger:
        """Alias for get_logger."""
        return cls.get_logger(name)


def get_audio_logger(name: str = "AudioAI") -> logging.Logger:
    """Helper function to retrieve configured audio logger instance."""
    return AudioLogger.get_logger(name)
