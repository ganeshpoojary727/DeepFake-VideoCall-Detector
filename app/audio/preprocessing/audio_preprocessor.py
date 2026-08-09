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
import soundfile as sf

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
    trim_silence : bool
        If True, apply silence trimming. Defaults to False for raw AASIST / ASVspoof pipelines.
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        trim_silence: bool = False,
    ) -> None:
        self.sample_rate = sample_rate or settings.audio.sample_rate
        self.trim_silence_flag = trim_silence

    # ── Public API ────────────────────────────

    def process(self, raw_input: str | Path) -> Tuple[np.ndarray, int]:
        """Alias for :meth:`preprocess` (satisfies ``BasePreprocessor``)."""
        return self.preprocess(raw_input)

    def preprocess(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline.

        Order:
            load → optional trim silence → peak normalize

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
        if self.trim_silence_flag:
            audio = self.trim_silence(audio)
        audio = self.normalize_audio(audio)
        return audio, sr

    # ── Individual steps ──────────────────────

    def load_audio(self, audio_path: str | Path) -> Tuple[np.ndarray, int]:
        """
        Load and resample an audio file using fast soundfile reader with librosa fallback.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if self.sample_rate is not None and sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
        except Exception:
            try:
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load audio file '{audio_path}': {exc}"
                ) from exc

        return audio, sr

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Peak-normalise the waveform to [-1, 1]."""
        max_val = float(np.max(np.abs(audio)))
        if max_val > 0.0:
            return audio / max_val
        return audio

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(audio)
        return trimmed