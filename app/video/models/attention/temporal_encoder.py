"""Production Temporal Encoder combining positional encoding, transformer blocks, and pooling."""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn

from app.video.configs.model_config import ModelConfig
from app.video.models.attention.base_attention import BaseTemporalAttention
from app.video.models.attention.positional_encoding import TemporalPositionalEncoding
from app.video.models.attention.temporal_pooling import TemporalPooling
from app.video.models.attention.transformer_block import TemporalTransformerBlock

logger = logging.getLogger(__name__)


class TemporalEncoder(BaseTemporalAttention):
    """Production Transformer-based temporal sequence encoder mapping (B, T, 1792) -> (B, 1792)."""

    def __init__(
        self,
        feature_dim: int = 1792,
        out_dim: int = 1792,
        config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(feature_dim=feature_dim, out_dim=out_dim)
        self.config = config or ModelConfig(feature_dim=feature_dim)

        num_heads = self.config.num_heads
        num_layers = self.config.num_layers
        attn_dropout = self.config.attn_dropout
        dropout = self.config.dropout
        pooling_type = self.config.pooling_type
        ffn_dim = self.config.ffn_dim

        self.pos_encoder = TemporalPositionalEncoding(
            max_len=128,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        self.pooling = TemporalPooling(feature_dim=feature_dim, pooling_type=pooling_type)

        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                activation_fn=self.config.activation_fn,
            )
            for _ in range(num_layers)
        ])

        self.out_proj = (
            nn.Linear(feature_dim, out_dim)
            if feature_dim != out_dim
            else nn.Identity()
        )

        logger.info(
            f"Initialized TemporalEncoder (feature_dim={feature_dim}, heads={num_heads}, "
            f"layers={num_layers}, pooling={pooling_type})"
        )

    def aggregate(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Aggregate frame sequence features (B, T, 1792) into single clip embedding (B, 1792).

        Args:
            sequence_features: Input frame embeddings tensor of shape (B, T, 1792).

        Returns:
            torch.Tensor: Single clip embedding of shape (B, 1792).
        """
        x = self.pos_encoder(sequence_features)
        x = self.pooling.prepend_cls_token(x)

        for block in self.blocks:
            x = block(x)

        clip_emb = self.pooling(x)
        return self.out_proj(clip_emb)

    def get_attention_weights(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """Extract attention weights over sequence frames.

        Args:
            sequence_features: Input frame embeddings tensor of shape (B, T, 1792).

        Returns:
            torch.Tensor: Attention weights of shape (B, T, 1) summing to 1.0 over T.
        """
        x = self.pos_encoder(sequence_features)
        x = self.pooling.prepend_cls_token(x)

        for block in self.blocks:
            x = block(x)

        return self.pooling.get_attention_weights(x)

