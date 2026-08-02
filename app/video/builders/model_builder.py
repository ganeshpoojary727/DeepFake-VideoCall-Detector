"""Video model builder module."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch.nn as nn

from app.video.configs.model_config import ModelConfig
from app.video.registry.model_registry import ModelRegistry, model_registry


class VideoModelBuilder:
    """Builder for constructing video neural network models from configuration."""

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self._registry = registry or model_registry

    @property
    def registry(self) -> ModelRegistry:
        """Get attached model registry instance."""
        return self._registry

    def build(
        self,
        config: Optional[ModelConfig] = None,
        override_params: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Construct PyTorch video neural network model from config object."""
        cfg = config or ModelConfig()
        model_name = cfg.model_name
        backbone_name = cfg.backbone_name

        # Lookup in registry by model_name or backbone_name
        model_key = backbone_name if backbone_name in self._registry.list_registered() else model_name
        
        try:
            model_cls = self._registry.get(model_key)
            kwargs: Dict[str, Any] = {}
            if override_params:
                kwargs.update(override_params)
            try:
                return model_cls(config=cfg, **kwargs)
            except Exception:
                return model_cls(num_classes=cfg.num_classes, in_channels=cfg.in_channels, **kwargs)
        except Exception:
            # Fallback to VideoFactory
            from app.video.models.video_factory import VideoFactory
            return VideoFactory.create_model(cfg)


# Alias class
ModelBuilder = VideoModelBuilder
