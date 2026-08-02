"""Base preprocessor interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
import torch


class BasePreprocessor(ABC):
    """Abstract base class for video frame preprocessors."""

    @abstractmethod
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Execute complete preprocessing workflow on input raw video or frame arrays.

        Args:
            input_data: Video file path, stream buffer, or list of frame arrays.

        Returns:
            torch.Tensor: Preprocessed PyTorch video tensor ready for model ingestion.
        """
        pass


# Base class alias
BaseVideoPreprocessor = BasePreprocessor
