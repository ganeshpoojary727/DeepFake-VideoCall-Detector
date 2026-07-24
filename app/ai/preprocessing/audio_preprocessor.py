"""
Audio preprocessing module.

This module is responsible for:
- Loading audio
- Resampling
- Normalizing
- Removing silence
"""

from pathlib import Path

import librosa
from networkx import add_path
import numpy as np

from app.config.settings import settings

class AudioPreprocessor:
    """
    Preprocess audio before feature extraction.
    """
    
    def __init__(self):

        self.sample_rate = settings.SAMPLE_RATE

    # --------------------------------------------------------
    # Load Audio
    # --------------------------------------------------------

    def load_audio(self, audio_path: str | Path):

     audio_path = Path(audio_path)

     if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

     audio, sample_rate = librosa.load(
        str(audio_path),
        sr=self.sample_rate
    )

     return audio, sample_rate

    # --------------------------------------------------------
    # Normalize Audio
    # --------------------------------------------------------

    def normalize_audio(self, audio: np.ndarray):

        return librosa.util.normalize(audio)

    # --------------------------------------------------------
    # Remove Silence
    # --------------------------------------------------------

    def trim_silence(self, audio: np.ndarray):

        audio, _ = librosa.effects.trim(audio)

        return audio

    # --------------------------------------------------------
    # Complete Pipeline
    # --------------------------------------------------------

    def preprocess(self, audio_path: str | Path):

        audio, sample_rate = self.load_audio(audio_path)

        audio = self.normalize_audio(audio)

        audio = self.trim_silence(audio)

        return audio, sample_rate