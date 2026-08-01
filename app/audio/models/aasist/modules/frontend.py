"""AASIST front-end feature extraction module.

Provides AASISTFrontEnd for processing raw audio signals through RawNetEncoder
and normalization layers into latent feature representations.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from app.audio.models.aasist.encoder import RawNetEncoder


class AASISTFrontEnd(nn.Module):
    """Front-end audio feature extraction module wrapping RawNetEncoder.

    Args:
        in_channels (int): Number of input audio channels (1 for mono).
        sinc_channels (int): Number of SincConv bandpass filters.
        res_channels (Optional[List[int]]): Residual block stack channel dimensions.
        embedding_dim (int): Output feature embedding dimension.
        sample_rate (int): Audio sampling rate in Hz.
    """

    def __init__(
        self,
        in_channels: int = 1,
        sinc_channels: int = 128,
        res_channels: Optional[List[int]] = None,
        embedding_dim: int = 128,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.encoder = RawNetEncoder(
            in_channels=in_channels,
            sinc_channels=sinc_channels,
            res_channels=res_channels,
            embedding_dim=embedding_dim,
            sample_rate=sample_rate,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract normalised latent feature representations from raw audio.

        Args:
            audio (torch.Tensor): Raw waveform tensor of shape (batch, 1, length) or (batch, length).

        Returns:
            torch.Tensor: Normalised feature embedding of shape (batch, embedding_dim).
        """
        features = self.encoder(audio)
        normalized = self.layer_norm(features)
        return normalized
