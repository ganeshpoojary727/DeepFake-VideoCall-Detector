"""Attention mechanism layers for audio feature maps.

Provides SqueezeExcitation (SE-Block) and SelfAttention1D modules.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block for 1D feature maps.

    Args:
        channels (int): Input channel dimension.
        reduction (int): Bottleneck channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying channel attention gating.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length).

        Returns:
            torch.Tensor: Channel-gated output tensor of shape (batch, channels, length).
        """
        batch_size, channels, _ = x.shape
        # Squeeze: Global Average Pooling across length
        squeezed = x.mean(dim=-1)

        # Excitation: Bottleneck MLP
        excitation = self.fc1(squeezed)
        excitation = self.act(excitation)
        excitation = self.fc2(excitation)
        weights = self.gate(excitation).unsqueeze(-1)

        # Re-weight channels
        return x * weights


class SelfAttention1D(nn.Module):
    """Scaled dot-product self-attention for 1D temporal feature sequences.

    Args:
        embed_dim (int): Feature embedding dimension per timestep.
        num_heads (int): Number of parallel attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch, embed_dim, length).

        Returns:
            torch.Tensor: Self-attended output tensor of shape (batch, embed_dim, length).
        """
        # Rearrange to (batch, length, embed_dim)
        seq = x.transpose(1, 2)
        batch_size, seq_len, _ = seq.shape

        q = self.q_proj(seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(seq).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Combine heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        # Rearrange back to (batch, embed_dim, length)
        return output.transpose(1, 2)
