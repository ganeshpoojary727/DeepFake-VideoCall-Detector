"""Weight and checkpoint loader utility module for PyTorch models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightLoader:
    """Utility class for loading pretrained ImageNet and custom checkpoint weights."""

    @staticmethod
    def load_weights(
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Load state dict weights into model from checkpoint file.

        Args:
            model: PyTorch module target.
            checkpoint_path: Path to .pt or .pth checkpoint.
            device: Map location target device.
            strict: Strict key matching enforcement.

        Returns:
            Dict[str, Any]: Loaded checkpoint dictionary or metadata.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        logger.info(f"Loading model weights from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Handle potential 'module.' prefix from DataParallel/DistributedDataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "").replace("net.", "")
            clean_state_dict[key] = v

        try:
            missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=strict)
            logger.info(
                f"Weights loaded successfully. (Missing keys: {len(missing_keys)}, "
                f"Unexpected keys: {len(unexpected_keys)})"
            )
        except Exception as err:
            logger.warning(f"Non-strict loading fallback engaged: {err}")
            model.load_state_dict(clean_state_dict, strict=False)

        return checkpoint if isinstance(checkpoint, dict) else {"state_dict": clean_state_dict}

    @staticmethod
    def load_onnx_weights_placeholder(onnx_path: str) -> None:
        """Placeholder hook for loading ONNX runtime inference sessions."""
        logger.info(f"ONNX loader placeholder invoked for: {onnx_path}")

    @staticmethod
    def load_tensorrt_engine_placeholder(engine_path: str) -> None:
        """Placeholder hook for loading TensorRT inference engines."""
        logger.info(f"TensorRT loader placeholder invoked for: {engine_path}")
