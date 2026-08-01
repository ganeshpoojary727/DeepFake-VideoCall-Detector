"""Base temporal attention module interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from app.video.registry.base_registry import BaseRegistry


class BaseTemporalAttention(nn.Module, ABC):
    """Abstract base class for temporal sequence aggregation attention modules."""

    def __init__(self, feature_dim: int = 1792, out_dim: int = 512) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self._out_dim = out_dim

    @property
    def feature_dim(self) -> int:
        """Get input feature dimension."""
        return self._feature_dim

    @property
    def output_dim(self) -> int:
        """Get output aggregated embedding dimension."""
        return self._out_dim

    @abstractmethod
    def aggregate(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Aggregate frame sequence features [B, T, D_in] -> [B, D_out]."""
        pass

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to aggregate method."""
        return self.aggregate(sequence_features)


class AttentionRegistry(BaseRegistry[BaseTemporalAttention]):
    """Registry for temporal attention architecture modules."""

    def __init__(self) -> None:
        super().__init__(name="AttentionRegistry")


# Global attention registry instance
attention_registry = AttentionRegistry()


class DummyTemporalAttention(BaseTemporalAttention):
    """Mock temporal attention implementation for infrastructure testing."""

    def __init__(self, feature_dim: int = 1792, out_dim: int = 512) -> None:
        super().__init__(feature_dim=feature_dim, out_dim=out_dim)
        self.proj = nn.Linear(feature_dim, out_dim)

    def aggregate(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Mean pool sequence and project to out_dim."""
        mean_feat = sequence_features.mean(dim=1)  # [B, feature_dim]
        return self.proj(mean_feat)


attention_registry.register("dummy_attention", DummyTemporalAttention, overwrite=True)
