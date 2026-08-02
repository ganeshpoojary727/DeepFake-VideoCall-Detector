"""Temporal attention package exports."""

from app.video.models.attention.base_attention import (
    BaseTemporalAttention,
    DummyTemporalAttention,
    AttentionRegistry,
    attention_registry,
)
from app.video.models.attention.positional_encoding import TemporalPositionalEncoding
from app.video.models.attention.multihead_attention import TemporalMultiHeadAttention
from app.video.models.attention.transformer_block import TemporalTransformerBlock
from app.video.models.attention.temporal_pooling import TemporalPooling
from app.video.models.attention.temporal_encoder import TemporalEncoder
from app.video.models.attention.temporal_feature_extractor import TemporalFeatureExtractor

# Register attention modules in registry
attention_registry.register("temporal_transformer", TemporalEncoder, overwrite=True)
attention_registry.register("temporal_encoder", TemporalEncoder, overwrite=True)
attention_registry.register("attention_pooling", TemporalEncoder, overwrite=True)

__all__ = [
    "BaseTemporalAttention",
    "DummyTemporalAttention",
    "AttentionRegistry",
    "attention_registry",
    "TemporalPositionalEncoding",
    "TemporalMultiHeadAttention",
    "TemporalTransformerBlock",
    "TemporalPooling",
    "TemporalEncoder",
    "TemporalFeatureExtractor",
]
