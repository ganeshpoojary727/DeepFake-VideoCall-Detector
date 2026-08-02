"""Base evaluator interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
    """Abstract base class for video model evaluation engines."""

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Execute evaluation over evaluation dataset split.

        Returns:
            Dict[str, Any]: Metrics dictionary (Accuracy, Precision, Recall, F1, ROC, AUC, EER, Latency, FPS).
        """
        pass


# Base class alias
BaseVideoEvaluator = BaseEvaluator
