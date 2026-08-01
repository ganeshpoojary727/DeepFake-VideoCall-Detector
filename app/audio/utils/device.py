"""
PyTorch hardware device selection and memory management utility.

Provides the DeviceManager class for querying available compute acceleration
(CUDA, MPS, CPU), memory tracking, and cache management.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from app.config.settings import settings


class DeviceManager:
    """
    Manager for selecting, inspecting, and managing PyTorch compute hardware.

    Parameters
    ----------
    requested_device : Optional[str | torch.device]
        Explicit device selection. Defaults to settings.DEVICE if None.
    """

    def __init__(self, requested_device: Optional[str | torch.device] = None) -> None:
        if requested_device is None:
            self._device = settings.DEVICE
        elif isinstance(requested_device, str):
            self._device = torch.device(requested_device)
        else:
            self._device = requested_device

    @property
    def device(self) -> torch.device:
        """Get current torch device instance."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if target device is CUDA GPU."""
        return self._device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if target device is Apple Silicon MPS."""
        return self._device.type == "mps"

    def to_device(self, tensor_or_model: Any) -> Any:
        """
        Move a tensor or model to the managed device.

        Parameters
        ----------
        tensor_or_model : Any
            PyTorch Tensor, Module, or nested dictionary/list of Tensors.

        Returns
        -------
        Any
            Object transferred to target compute device.
        """
        if hasattr(tensor_or_model, "to"):
            return tensor_or_model.to(self._device)
        if isinstance(tensor_or_model, dict):
            return {k: self.to_device(v) for k, v in tensor_or_model.items()}
        if isinstance(tensor_or_model, (list, tuple)):
            return [self.to_device(v) for v in tensor_or_model]
        return tensor_or_model

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current GPU memory usage stats in megabytes (MB).

        Returns
        -------
        Dict[str, float]
            Dictionary containing allocated, reserved, and peak memory in MB.
        """
        if not self.is_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

        bytes_in_mb = 1024 * 1024
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(self._device) / bytes_in_mb, 2),
            "reserved_mb": round(torch.cuda.memory_reserved(self._device) / bytes_in_mb, 2),
            "max_allocated_mb": round(
                torch.cuda.max_memory_allocated(self._device) / bytes_in_mb, 2
            ),
        }

    def clear_cache(self) -> None:
        """Release unused GPU memory cached by PyTorch allocator."""
        if self.is_cuda:
            torch.cuda.empty_cache()
