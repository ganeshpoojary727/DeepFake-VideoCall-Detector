"""Base model interface specification.

Provides the BaseAudioModel abstract class that all audio neural network models
must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseAudioModel(ABC, nn.Module):
    """Abstract base class for all audio deepfake detection neural network architectures."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, ...).

        Returns:
            torch.Tensor: Model logits of shape (batch_size, num_classes).
        """
        pass

    @abstractmethod
    def get_num_parameters(self) -> int:
        """Calculate total number of trainable model parameters.

        Returns:
            int: Number of trainable parameters.
        """
        pass

    def save(self, path: Path | str) -> None:
        """Serialize model weights to disk.

        Args:
            path (Path | str): Target file path for the checkpoint.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target_path)

    def load(self, path: Path | str, device: Optional[torch.device] = None) -> None:
        """Load model weights from disk checkpoint.

        Args:
            path (Path | str): Path to saved checkpoint file.
            device (Optional[torch.device]): Computing device mapping.
        """
        checkpoint_path = Path(path)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
