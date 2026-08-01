"""
Centralised logging for the DeepFake Video Call Detector.

Features
────────
• Module-level loggers via ``get_logger(__name__)``
• RotatingFileHandler  (10 MB × 5 backups)
• Console + file dual output
• Consistent format with module name, timestamp, and level
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config.settings import settings


_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _configure_root_logger() -> None:
    """One-time setup of the root 'DeepFakeDetector' logger hierarchy."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("DeepFakeDetector")
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Rotating file handler ────────────────
    log_path = settings.LOG_DIR / "application.log"
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # ── Console handler ──────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str = "DeepFakeDetector") -> logging.Logger:
    """
    Return a logger scoped under the project root logger.

    Usage::

        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Training started")

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.  Loggers are arranged
        hierarchically under ``DeepFakeDetector``.
    """
    _configure_root_logger()

    # Nest under the project root so handlers propagate
    if not name.startswith("DeepFakeDetector"):
        name = f"DeepFakeDetector.{name}"

    return logging.getLogger(name)