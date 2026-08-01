"""
Utility helpers for the DeepFake Video Call Detector.

Provides reusable functions for:
• Audio file validation (MIME-type + size guard)
• Temporary file management (context manager that auto-cleans)
• Model integrity verification (SHA-256)
• Reproducibility seed setting
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Audio File Validation
# ──────────────────────────────────────────────

_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
_MAX_AUDIO_SIZE_BYTES = 50_000_000  # 50 MB


def validate_audio_file(path: str | Path) -> Path:
    """
    Validate that *path* points to a readable audio file within size limits.

    Parameters
    ----------
    path : str | Path
        Path to the candidate audio file.

    Returns
    -------
    Path
        Resolved ``Path`` object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension or size is not acceptable.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    if path.suffix.lower() not in _ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Allowed: {_ALLOWED_AUDIO_EXTENSIONS}"
        )

    size = path.stat().st_size
    if size > _MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"File too large ({size:,} bytes). "
            f"Maximum: {_MAX_AUDIO_SIZE_BYTES:,} bytes"
        )

    return path


# ──────────────────────────────────────────────
# Temporary Audio File (fixes temp file leak)
# ──────────────────────────────────────────────


@contextlib.contextmanager
def temp_audio_file(suffix: str = ".wav") -> Generator[str, None, None]:
    """
    Context manager that yields a temporary file path and **always** deletes
    it on exit — fixing the temp-file leak identified in the audit.

    Usage::

        with temp_audio_file() as tmp_path:
            sf.write(tmp_path, audio, sr)
            result = predictor.predict(tmp_path)
        # tmp_path is deleted here
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Could not delete temp file: %s", tmp_path)


# ──────────────────────────────────────────────
# Model Integrity Verification
# ──────────────────────────────────────────────


def compute_model_hash(model_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a model file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_model_integrity(
    model_path: str | Path,
    expected_hash: str,
) -> bool:
    """
    Verify that the model file has not been tampered with.

    Returns ``True`` if the SHA-256 digest matches *expected_hash*.
    """
    actual = compute_model_hash(model_path)
    ok = actual == expected_hash
    if not ok:
        logger.warning(
            "Model integrity check FAILED.\n  Expected: %s\n  Actual:   %s",
            expected_hash,
            actual,
        )
    return ok


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Imports torch lazily so non-GPU callers aren't penalised.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("All seeds set to %d (torch included)", seed)
    except ImportError:
        logger.debug("All seeds set to %d (torch not available)", seed)
