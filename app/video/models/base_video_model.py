"""Base video deepfake detector model abstraction module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from app.video.configs.model_config import ModelConfig


class BaseVideoModel(nn.Module, ABC):
    """Abstract base class for all video deepfake detector architectures.

    Accepts video tensors of shape [B, T, C, H, W] and computes classification logits [B, num_classes].
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

    @property
    def num_classes(self) -> int:
        """Get output logits count."""
        return self.config.num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input video sequence tensor [B, T, C, H, W].

        Returns:
            torch.Tensor: Classification logits [B, num_classes].
        """
        pass

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Optional feature extraction interface. Default returns forward logits."""
        return self.forward(x)

    def get_num_parameters(self) -> Tuple[int, int]:
        """Compute total and trainable model parameter counts.

        Returns:
            Tuple[int, int]: (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
