"""Modular classification head for neural network backbones."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class ModularClassifierHead(nn.Module):
    """Modular classification head with dropout, normalization, activation, and linear projection."""

    def __init__(
        self,
        in_features: int = 1792,
        num_classes: int = 2,
        dropout: float = 0.2,
        activation_fn: str = "silu",
        norm_layer: str = "identity",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Normalization
        norm_key = norm_layer.lower().strip()
        if norm_key == "batchnorm":
            self.norm: nn.Module = nn.BatchNorm1d(in_features)
        elif norm_key == "layernorm":
            self.norm = nn.LayerNorm(in_features)
        else:
            self.norm = nn.Identity()

        # Activation
        act_key = activation_fn.lower().strip()
        if act_key == "silu":
            self.act: nn.Module = nn.SiLU(inplace=True)
        elif act_key == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_key == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor of shape (B, in_features) or (B, in_features, 1, 1).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        if x.ndim == 4:
            x = x.squeeze(-1).squeeze(-1)
        elif x.ndim > 2:
            x = x.view(x.size(0), -1)

        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc(x)
