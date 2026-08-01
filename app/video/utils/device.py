"""Device detection and management utility module."""

from __future__ import annotations

import torch


def get_device(preferred_device: str = "cuda") -> torch.device:
    """Select target PyTorch compute device based on system hardware availability.

    Args:
        preferred_device: Desired target device string.

    Returns:
        torch.device: Available PyTorch device object.
    """
    if preferred_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred_device)
    if preferred_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeviceManager:
    """Device manager class encapsulating device state."""

    def __init__(self, preferred: str = "cuda") -> None:
        self._device = get_device(preferred)

    @property
    def device(self) -> torch.device:
        """Get target torch device instance."""
        return self._device

    def to_device(self, tensor_or_module: torch.Tensor | torch.nn.Module) -> torch.Tensor | torch.nn.Module:
        """Move tensor or nn.Module to managed device."""
        return tensor_or_module.to(self._device)
