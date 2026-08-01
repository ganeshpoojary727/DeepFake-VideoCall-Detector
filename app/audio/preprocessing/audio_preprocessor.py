"""
Audio preprocessing pipeline.

Responsibilities
────────────────
• Load audio files (any librosa-supported format)
• Trim silence **before** normalisation (fixes audit pipeline order)
• Peak-normalise the waveform
• Validate inputs and handle errors gracefully

Removed
───────
• Dead ``from networkx import add_path`` import (audit §2.3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

from app.config.settings import settings
from app.core.interfaces import BasePreprocessor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AudioPreprocessor(BasePreprocessor):
    """
    Preprocess raw audio files into clean waveforms ready for feature
    extraction.

    Parameters
    ----------
    sample_rate : int | None
        Target sample rate.  Defaults to ``settings.audio.sample_rate``.
    """

    def __init__(self, sample_rate: int | None = None) -> None:
        self.sample_rate = sample_rate or settings.audio.sample_rate

    # ── Public API ────────────────────────────

    def process(self, raw_input: str | Path) -> Tuple[np.ndarray, int]:
        """Alias for :meth:`preprocess` (satisfies ``BasePreprocessor``)."""
        return self.preprocess(raw_input)

    def preprocess(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline.

        Order (fixed per audit §5.2):
            load → trim silence → normalise

        Parameters
        ----------
        audio_path : str | Path
            Path to the audio file.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(waveform, sample_rate)``
        """
        audio, sr = self.load_audio(audio_path)
        audio = self.trim_silence(audio)
        audio = self.normalize_audio(audio)
        return audio, sr

    # ── Individual steps ──────────────────────

    def load_audio(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load and resample an audio file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        RuntimeError
            If librosa cannot decode the file.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file '{audio_path}': {exc}"
            ) from exc

        return audio, sr

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalise the waveform to [-1, 1]."""
        return librosa.util.normalize(audio)

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(audio)
        return trimmed