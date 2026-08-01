"""
Video deepfake detection model — EfficientNet-style lightweight CNN.

Architecture
------------
An EfficientNet-inspired face image classifier:

    Stem: Conv(3→32, 3×3) → BN → ReLU
    Stage 1: MBConv(32→16, exp=1)
    Stage 2: MBConv(16→24, exp=6) × 2
    Stage 3: MBConv(24→40, exp=6) × 2
    Stage 4: MBConv(40→80, exp=6) × 3
    Head: AdaptiveAvgPool → Linear(80, 2)

Input: (batch, 3, 224, 224) normalised RGB face image
Output: (batch, 2) logits [bonafide, spoof]

Parameters: ~1.2M — fast inference on CPU for real-time video calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# MobileNet-V2 Inverted Residual Block
# ──────────────────────────────────────────────


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution block (MBConv)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expansion: int = 6,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expand
        if expansion > 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ]
        # Project
        layers += [
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.drop(out)
        if self.use_residual:
            return x + out
        return out


# ──────────────────────────────────────────────
# VideoDeepFakeCNN
# ──────────────────────────────────────────────


class VideoDeepFakeCNN(nn.Module):
    """
    Lightweight EfficientNet-style CNN for face-based deepfake detection.

    Parameters
    ----------
    num_classes : int
        Output classes (default 2: bonafide / spoof).
    in_channels : int
        Input channels (default 3 for RGB).
    dropout : float
        Classifier dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Backbone stages
        self.stages = nn.Sequential(
            # Stage 1 — no expansion, stride=1
            MBConv(32, 16, expansion=1, stride=1, dropout=0.1),
            # Stage 2 — 2 blocks
            MBConv(16, 24, expansion=6, stride=2, dropout=0.1),
            MBConv(24, 24, expansion=6, stride=1, dropout=0.1),
            # Stage 3 — 2 blocks
            MBConv(24, 40, expansion=6, stride=2, dropout=0.2),
            MBConv(40, 40, expansion=6, stride=1, dropout=0.2),
            # Stage 4 — 3 blocks
            MBConv(40, 80, expansion=6, stride=2, dropout=0.2),
            MBConv(80, 80, expansion=6, stride=1, dropout=0.2),
            MBConv(80, 80, expansion=6, stride=1, dropout=0.2),
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(80, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.classifier(x)
