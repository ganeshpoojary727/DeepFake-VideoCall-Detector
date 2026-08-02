"""Video model registry module."""

from __future__ import annotations

from typing import Dict, Type
import torch.nn as nn

from app.video.models.base_video_model import BaseVideoModel
from app.video.registry.base_registry import BaseRegistry


class VideoModelRegistry(BaseRegistry[nn.Module]):
    """Registry for video neural network model architectures."""

    def __init__(self) -> None:
        super().__init__(name="VideoModelRegistry")


# Class alias
ModelRegistry = VideoModelRegistry

# Global model registry instance
model_registry = VideoModelRegistry()
