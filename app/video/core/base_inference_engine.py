"""Base inference engine interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
import numpy as np


class BaseInferenceEngine(ABC):
    """Abstract base class for real-time video deepfake inference engines."""

    @abstractmethod
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Predict deepfake score for single video frame array.

        Args:
            frame: RGB numpy array [H, W, 3].

        Returns:
            Dict[str, Any]: Prediction dictionary containing confidence score, label, and latency.
        """
        pass

    @abstractmethod
    def predict_sequence(self, frames: List[np.ndarray] | torch.Tensor) -> Dict[str, Any]:
        """Predict deepfake score for temporal frame sequence tensor or frame list.

        Args:
            frames: Sequence of frame arrays or 5D PyTorch tensor [B, T, C, H, W].

        Returns:
            Dict[str, Any]: Prediction dictionary containing sequence score, confidence, and metadata.
        """
        pass


# Base class alias
BaseVideoInferenceEngine = BaseInferenceEngine
