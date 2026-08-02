"""Feature extractor registry module."""

from __future__ import annotations

from app.video.features.base_feature_extractor import BaseFeatureExtractor
from app.video.registry.base_registry import BaseRegistry


class FeatureExtractorRegistry(BaseRegistry[BaseFeatureExtractor]):
    """Registry for feature extractor classes."""

    def __init__(self) -> None:
        super().__init__(name="FeatureExtractorRegistry")


# Class alias
FeatureRegistry = FeatureExtractorRegistry

# Global feature extractor registry instance
feature_registry = FeatureExtractorRegistry()
