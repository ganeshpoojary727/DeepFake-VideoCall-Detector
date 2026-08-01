"""Base predictor interface specification.

Provides the BaseAudioPredictor abstract class for real-time and batch audio inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from app.core.interfaces import PredictionResult


class BaseAudioPredictor(ABC):
    """Abstract base class for audio deepfake inference predictors."""

    @abstractmethod
    def predict(self, audio_input: Union[str, Path, Any]) -> PredictionResult:
        """Run deepfake prediction on single audio file or waveform.

        Args:
            audio_input (Union[str, Path, Any]): Path to audio file or raw array.

        Returns:
            PredictionResult: Prediction result container (label, confidence, latency).
        """
        pass

    @abstractmethod
    def predict_batch(
        self, audio_inputs: List[Union[str, Path, Any]]
    ) -> List[PredictionResult]:
        """Run batch deepfake prediction on multiple audio inputs.

        Args:
            audio_inputs (List[Union[str, Path, Any]]): List of audio file paths or arrays.

        Returns:
            List[PredictionResult]: List of PredictionResult objects.
        """
        pass
