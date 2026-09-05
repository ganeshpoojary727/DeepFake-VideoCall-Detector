"""Base video model interface specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class BaseVideoModel(ABC, nn.Module):
    """Abstract base class for all video deepfake detection neural network architectures."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass over video frame sequence tensor.

        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, sequence_length, channels, height, width)
               or (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Model logit output tensor of shape (batch_size, num_classes).
        """
        pass

    @abstractmethod
    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters in model.

        Returns:
            int: Trainable parameter count.
        """
        pass

    def save(self, path: Path | str) -> None:
        """Serialize model state dict to disk checkpoint.

        Args:
            path: Target file path.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target_path)

    def load(self, path: Path | str, device: Optional[torch.device] = None) -> None:
        """Load model state dict from saved checkpoint file.

        Args:
            path: Target checkpoint path.
            device: Optional computing device mapping.
        """
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(payload, dict):
            state_dict = payload.get("model_state", payload.get("state_dict", payload.get("model", payload)))
        else:
            state_dict = payload
        clean_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        self.load_state_dict(clean_dict)


# Base class alias
BaseModel = BaseVideoModel
