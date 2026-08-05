"""RawNet-style feature encoder module for audio representation extraction.

Provides RawNetEncoder for extracting latent feature embeddings from raw waveforms
or spectrograms using SincConv filterbanks, 1D residual blocks, and SE attention.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from app.audio.models.layers.attention import SqueezeExcitation
from app.audio.models.layers.conv import Conv1DBlock, SincConv
from app.audio.models.layers.residual import ResidualBlock1D


class RawNetEncoder(nn.Module):
    """RawNet-style front-end encoder for extracting latent audio embeddings.

    Args:
        in_channels (int): Input channel dimension (1 for raw audio or spectrograms).
        sinc_channels (int): Number of SincConv bandpass filters.
        res_channels (List[int]): Channel dimensions for residual block stack.
        embedding_dim (int): Output latent embedding vector dimension.
        sample_rate (int): Sampling rate for SincConv initialization.
        use_checkpointing (bool): Enable gradient/activation checkpointing during training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        sinc_channels: int = 128,
        res_channels: Optional[List[int]] = None,
        embedding_dim: int = 128,
        sample_rate: int = 16000,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing
        if res_channels is None:
            res_channels = [128, 128, 256, 256, 512, 512]

        self.sinc_conv = SincConv(
            out_channels=sinc_channels,
            kernel_size=251,
            sample_rate=sample_rate,
        )
        self.first_bn = nn.BatchNorm1d(sinc_channels)
        self.first_act = nn.LeakyReLU(0.2, inplace=True)

        # Build stack of 1D residual blocks with SE attention
        res_layers: List[nn.Module] = []
        curr_channels = sinc_channels

        for i, out_c in enumerate(res_channels):
            downsample = i % 2 == 1  # Downsample every 2nd block
            res_layers.append(
                ResidualBlock1D(
                    in_channels=curr_channels,
                    out_channels=out_c,
                    downsample=downsample,
                )
            )
            res_layers.append(SqueezeExcitation(channels=out_c, reduction=8))
            curr_channels = out_c

        self.res_stack = nn.Sequential(*res_layers)
        self.out_bn = nn.BatchNorm1d(curr_channels)
        self.out_act = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embedding = nn.Linear(curr_channels, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent audio feature embeddings.

        Args:
            x (torch.Tensor): Raw waveform or spectrogram tensor of shape (batch, 1, length).

        Returns:
            torch.Tensor: Latent audio embedding vectors of shape (batch, embedding_dim).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.sinc_conv(x)
        x = self.first_bn(x)
        x = self.first_act(x)

        if self.training and self.use_checkpointing:
            for layer in self.res_stack:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            x = self.res_stack(x)

        x = self.out_bn(x)
        x = self.out_act(x)

        # Global pooling and embedding projection
        x = self.pool(x).squeeze(-1)
        embedding = self.fc_embedding(x)
        return embedding
