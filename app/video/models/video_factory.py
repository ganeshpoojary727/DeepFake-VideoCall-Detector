"""Video model factory module."""

from __future__ import annotations

from typing import Optional, Type
import torch
import torch.nn as nn

from app.video.configs.model_config import ModelConfig
from app.video.exceptions.video_exceptions import ModelError
from app.video.models.attention.base_attention import (
    BaseTemporalAttention,
    DummyTemporalAttention,
    attention_registry,
)
from app.video.models.backbones.base_backbone import (
    BaseBackbone,
    DummyBackbone,
    backbone_registry,
)
from app.video.models.base_video_model import BaseVideoModel
from app.video.models.classifiers.base_classifier import (
    BaseClassifier,
    LinearClassifier,
    classifier_registry,
)
from app.video.models.efficientnet.backbone import EfficientNetB4Backbone
from app.video.models.efficientnet.model import EfficientNetB4Model
from app.video.models.model_registry import model_registry


class ModularVideoModel(BaseVideoModel):
    """Assembles a video detector model from backbone, attention, and classifier components."""

    def __init__(
        self,
        backbone: BaseBackbone,
        attention: BaseTemporalAttention,
        classifier: BaseClassifier,
        config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(config=config)
        self.backbone = backbone
        self.attention = attention
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass [B, T, C, H, W] -> [B, num_classes]."""
        if x.dim() == 4:
            x = x.unsqueeze(0)

        b, t, c, h, w = x.shape
        frames_flat = x.view(b * t, c, h, w)

        spatial_feats = self.backbone(frames_flat)
        spatial_feats = spatial_feats.view(b, t, -1)

        temp_feats = self.attention(spatial_feats)
        logits = self.classifier(temp_feats)
        return logits


class VideoFactory:
    """Factory for building complete Video AI model instances from config."""

    @classmethod
    def create_model(cls, config: Optional[ModelConfig] = None) -> BaseVideoModel:
        """Create video model from ModelConfig."""
        cfg = config or ModelConfig()
        cfg.validate()

        b_name = cfg.backbone_name.lower().strip()
        m_name = cfg.model_name.lower().strip()

        if b_name == "efficientnet_b4" or m_name == "efficientnet_b4":
            return EfficientNetB4Model(config=cfg)

        # Fallback to modular assembly
        try:
            backbone_cls = backbone_registry.get(cfg.backbone_name)
            backbone = backbone_cls(in_channels=cfg.in_channels, feature_dim=cfg.feature_dim)
        except Exception:
            backbone = DummyBackbone(in_channels=cfg.in_channels, feature_dim=cfg.feature_dim)

        att_name = cfg.attention_name or "dummy_attention"
        try:
            attention_cls = attention_registry.get(att_name)
            attention = attention_cls(feature_dim=cfg.feature_dim, out_dim=512)
        except Exception:
            attention = DummyTemporalAttention(feature_dim=cfg.feature_dim, out_dim=512)

        try:
            classifier_cls = classifier_registry.get(cfg.classifier_name)
            classifier = classifier_cls(in_features=512, num_classes=cfg.num_classes, dropout=cfg.dropout)
        except Exception:
            classifier = LinearClassifier(in_features=512, num_classes=cfg.num_classes, dropout=cfg.dropout)

        return ModularVideoModel(
            backbone=backbone,
            attention=attention,
            classifier=classifier,
            config=cfg,
        )


# Register models into global registry
model_registry.register("modular_video_model", ModularVideoModel, overwrite=True)
model_registry.register("efficientnet_b4", EfficientNetB4Model, overwrite=True)
model_registry.register("efficientnet_b0", EfficientNetB4Model, overwrite=True)
model_registry.register("efficientnet_b2", EfficientNetB4Model, overwrite=True)
model_registry.register("efficientnet_b5", EfficientNetB4Model, overwrite=True)
backbone_registry.register("efficientnet_b4", EfficientNetB4Backbone, overwrite=True)
