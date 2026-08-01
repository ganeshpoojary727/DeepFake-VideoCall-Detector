"""Convolutional building block layers for audio neural networks.

Provides Conv1DBlock, Conv2DBlock, and SincConv parametric filterbank layers.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """1D Convolution block with Batch Normalization, Activation, and Dropout.

    Args:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension.
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Padding size.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, new_length).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class Conv2DBlock(nn.Module):
    """2D Convolution block with Batch Normalization, Activation, and Pooling.

    Args:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimension.
        kernel_size (int | tuple): Convolution kernel size.
        stride (int | tuple): Convolution stride.
        padding (int | tuple): Padding size.
        pool_size (Optional[int | tuple]): MaxPool2d kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        pool_size: Optional[int | tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, H_out, W_out).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class SincConv(nn.Module):
    """Sinc-convolution layer (SincNet) for raw audio signal bandpass filtering.

    Args:
        out_channels (int): Number of bandpass filters.
        kernel_size (int): Filter kernel length (must be odd).
        sample_rate (int): Audio sampling rate in Hz.
    """

    def __init__(
        self,
        out_channels: int = 128,
        kernel_size: int = 251,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize bandpass frequencies on Mel scale
        min_freq = 30.0
        max_freq = sample_rate / 2.0 - 100.0

        min_mel = 2595.0 * np.log10(1.0 + min_freq / 700.0)
        max_mel = 2595.0 * np.log10(1.0 + max_freq / 700.0)
        mels = np.linspace(min_mel, max_mel, out_channels + 1)
        freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        self.freq_low = nn.Parameter(torch.from_numpy(freqs[:-1]).float())
        self.freq_band = nn.Parameter(torch.from_numpy(np.diff(freqs)).float())

        n = (self.kernel_size - 1) / 2.0
        self.window = 0.54 - 0.46 * torch.cos(
            2.0 * math.pi * torch.arange(-n, n + 1) / self.kernel_size
        )
        n_grid = 2.0 * math.pi * torch.arange(-n, n + 1) / sample_rate
        self.register_buffer("n_grid", n_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying bandpass Sinc filters.

        Args:
            x (torch.Tensor): Raw waveform tensor of shape (batch, 1, length).

        Returns:
            torch.Tensor: Filtered feature map of shape (batch, out_channels, length_out).
        """
        f_low = torch.abs(self.freq_low)
        f_high = torch.clamp(f_low + torch.abs(self.freq_band), max=self.sample_rate / 2.0)

        f_low_mat = f_low.unsqueeze(1)
        f_high_mat = f_high.unsqueeze(1)
        n_mat = self.n_grid.unsqueeze(0)

        # Compute ideal bandpass sinc filters
        sinc_high = torch.sin(f_high_mat * n_mat) / (n_mat / 2.0 + 1e-8)
        sinc_low = torch.sin(f_low_mat * n_mat) / (n_mat / 2.0 + 1e-8)
        bandpass = (sinc_high - sinc_low) * self.window.to(x.device).unsqueeze(0)

        # Normalize energy per filter
        bandpass = bandpass / (2.0 * (f_high_mat - f_low_mat) + 1e-8)
        filters = bandpass.unsqueeze(1)

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)
