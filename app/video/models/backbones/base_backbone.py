"""Base spatial backbone interface and registry module."""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from app.video.registry.base_registry import BaseRegistry


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for 2D frame spatial feature extractors (e.g. EfficientNet)."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 1792) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._feature_dim = feature_dim

    @property
    def in_channels(self) -> int:
        """Get input image channel count."""
        return self._in_channels

    @property
    def feature_dim(self) -> int:
        """Get output spatial embedding dimensionality."""
        return self._feature_dim

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features [B*T, C, H, W] -> [B*T, feature_dim]."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding to extract_features."""
        return self.extract_features(x)


class BackboneRegistry(BaseRegistry[BaseBackbone]):
    """Registry for backbone architecture implementations."""

    def __init__(self) -> None:
        super().__init__(name="BackboneRegistry")


# Global backbone registry instance
backbone_registry = BackboneRegistry()


class DummyBackbone(BaseBackbone):
    """Mock backbone implementation for testing infrastructure contracts."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 1792) -> None:
        super().__init__(in_channels=in_channels, feature_dim=feature_dim)
        self.proj = nn.AdaptiveAvgPool2d((1, 1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return dummy spatial feature embedding [B*T, feature_dim]."""
        b_t = x.shape[0]
        device = x.device
        dtype = x.dtype
        out = torch.zeros((b_t, self._feature_dim), device=device, dtype=dtype)
        return out


backbone_registry.register("dummy_backbone", DummyBackbone, overwrite=True)
