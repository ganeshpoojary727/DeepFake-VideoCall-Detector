"""Exponential Moving Average (EMA) model wrapper for training stability."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any
import torch
import torch.nn as nn


class EMAModel:
    """Maintains Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow_model = deepcopy(model)
        self.shadow_model.eval()
        for p in self.shadow_model.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        """Update shadow model weights using exponential moving average."""
        with torch.no_grad():
            msd = model.state_dict()
            for key, param in self.shadow_model.state_dict().items():
                if key in msd:
                    if param.dtype.is_floating_point:
                        param.copy_(self.decay * param + (1.0 - self.decay) * msd[key].to(param.device))
                    else:
                        param.copy_(msd[key])

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict of shadow model."""
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict into shadow model."""
        self.shadow_model.load_state_dict(state_dict)
