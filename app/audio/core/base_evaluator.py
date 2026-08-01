"""Base evaluator interface specification.

Provides the BaseAudioEvaluator abstract class for model evaluation engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAudioEvaluator(ABC):
    """Abstract base class for audio deepfake model evaluator implementations."""

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Execute evaluation over designated evaluation dataset split.

        Returns:
            Dict[str, Any]: Evaluated metrics dictionary (Accuracy, EER, minDCF, confusion matrix).
        """
        pass
