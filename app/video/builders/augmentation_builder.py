"""Video augmentation pipeline builder module."""

from __future__ import annotations

from typing import Optional
from app.video.augmentation.augmentation_pipeline import VideoAugmentationPipeline
from app.video.configs.augmentation_config import AugmentationConfig


class AugmentationBuilder:
    """Builder for constructing video data augmentation pipelines from AugmentationConfig."""

    def build(self, config: Optional[AugmentationConfig] = None) -> VideoAugmentationPipeline:
        """Construct VideoAugmentationPipeline from configuration.

        Args:
            config: Augmentation configuration object.

        Returns:
            VideoAugmentationPipeline: Composed augmentation pipeline.
        """
        cfg = config or AugmentationConfig()
        return VideoAugmentationPipeline(config=cfg)
