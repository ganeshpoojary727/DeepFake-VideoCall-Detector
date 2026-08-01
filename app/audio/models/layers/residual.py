"""Residual building block layers for 1D and 2D audio feature maps.

Provides ResidualBlock1D and ResidualBlock2D classes with shortcut projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """1D Residual block with optional shortcut projection and downsampling.

    Args:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension.
        stride (int): Convolution stride for downsampling.
        downsample (bool): Whether to downsample feature map length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.pool = nn.MaxPool1d(2) if downsample else nn.Identity()

        # Shortcut projection if dimensions change or downsampling is requested
        if in_channels != out_channels or stride != 1 or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2) if downsample else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, length).

        Returns:
            torch.Tensor: Residual output tensor of shape (batch, out_channels, length_out).
        """
        residual = self.shortcut(x)

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.pool(out)
        return out + residual


class ResidualBlock2D(nn.Module):
    """2D Residual block with optional shortcut projection.

    Args:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension.
        stride (int | tuple): Convolution stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).

        Returns:
            torch.Tensor: Residual output tensor of shape (batch, out_channels, H_out, W_out).
        """
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        return self.act2(out)
