"""Base preprocessor interface specification.

Provides the BaseAudioPreprocessor abstract class for raw audio signal loaders and cleaning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

import numpy as np


class BaseAudioPreprocessor(ABC):
    """Abstract base class for audio signal preprocessing implementations."""

    @abstractmethod
    def preprocess(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Preprocess audio file from disk into clean waveform.

        Args:
            audio_path (Union[str, Path]): Path to target audio file.

        Returns:
            Tuple[np.ndarray, int]: Waveform numpy array and sampling rate.
        """
        pass

    @abstractmethod
    def load_audio(self, path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file into raw floating point waveform array.

        Args:
            path (Union[str, Path]): Path to target audio file.

        Returns:
            Tuple[np.ndarray, int]: Raw waveform array and sampling rate.
        """
        pass

    @abstractmethod
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Apply amplitude normalization to audio sample array.

        Args:
            audio (np.ndarray): Input audio waveform array.

        Returns:
            np.ndarray: Amplitude-normalized audio array.
        """
        pass
