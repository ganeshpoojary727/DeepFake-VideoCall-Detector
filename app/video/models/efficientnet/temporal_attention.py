"""EfficientNet subsystem temporal attention module connector."""

from __future__ import annotations

from app.video.models.attention.temporal_encoder import TemporalEncoder
from app.video.models.attention.temporal_feature_extractor import TemporalFeatureExtractor

# Subsystem module aliases
TemporalAttention = TemporalEncoder

__all__ = [
    "TemporalAttention",
    "TemporalEncoder",
    "TemporalFeatureExtractor",
]
