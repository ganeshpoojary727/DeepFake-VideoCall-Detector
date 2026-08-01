"""Base classification head interface and registry module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from app.video.registry.base_registry import BaseRegistry


class BaseClassifier(nn.Module, ABC):
    """Abstract base class for classification output heads."""

    def __init__(self, in_features: int = 512, num_classes: int = 2) -> None:
        super().__init__()
        self._in_features = in_features
        self._num_classes = num_classes

    @property
    def in_features(self) -> int:
        """Get input feature dimension."""
        return self._in_features

    @property
    def num_classes(self) -> int:
        """Get output logits count."""
        return self._num_classes

    @abstractmethod
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Map feature embeddings [B, in_features] to logits [B, num_classes]."""
        pass

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to classify."""
        return self.classify(features)


class ClassifierRegistry(BaseRegistry[BaseClassifier]):
    """Registry for classification head implementations."""

    def __init__(self) -> None:
        super().__init__(name="ClassifierRegistry")


# Global classifier registry instance
classifier_registry = ClassifierRegistry()


class LinearClassifier(BaseClassifier):
    """Simple linear classification head implementation."""

    def __init__(self, in_features: int = 512, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__(in_features=in_features, num_classes=num_classes)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits."""
        return self.fc(self.drop(features))


classifier_registry.register("linear", LinearClassifier, overwrite=True)
