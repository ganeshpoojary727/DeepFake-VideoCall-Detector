"""Video model serialization and checkpoint loading module."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from app.video.exceptions.video_exceptions import ModelError
from app.video.utils.checkpoint_utils import load_checkpoint, save_checkpoint


class VideoModelLoader:
    """Handles saving and loading PyTorch video model weights and state."""

    @staticmethod
    def save_model(
        model: nn.Module,
        filepath: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model state dict to file.

        Args:
            model: PyTorch video model module.
            filepath: Target output file path.
            extra_meta: Optional metadata dict.
        """
        try:
            save_checkpoint(filepath=filepath, model=model, extra_meta=extra_meta)
        except Exception as err:
            raise ModelError(f"Failed to save model checkpoint to {filepath}") from err

    @staticmethod
    def load_model(
        model: nn.Module,
        filepath: str,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load model state dict from checkpoint path.

        Args:
            model: PyTorch model module instance.
            filepath: Checkpoint file path.
            device: Target torch device string.
            strict: Whether to enforce strict key matching.

        Returns:
            Dict[str, Any]: Checkpoint payload dict.
        """
        if not os.path.exists(filepath):
            raise ModelError(f"Model checkpoint file not found: {filepath}")

        try:
            payload = load_checkpoint(filepath=filepath, model=model, device=device)
            return payload
        except Exception as err:
            raise ModelError(f"Failed to load model from {filepath}") from err
