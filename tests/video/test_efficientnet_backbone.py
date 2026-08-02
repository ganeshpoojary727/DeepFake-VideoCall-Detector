"""Exhaustive unit tests for production EfficientNet-B4 Backbone subsystem."""

from pathlib import Path
import pytest
import torch

from app.video.builders import VideoModelBuilder
from app.video.configs import ModelConfig
from app.video.models import (
    EfficientNetB4Backbone,
    EfficientNetB4Model,
    ExecutionMode,
    FeatureExtractor,
    ModularClassifierHead,
    WeightLoader,
    model_registry,
)


def test_backbone_forward_and_feature_extraction():
    backbone = EfficientNetB4Backbone(pretrained=False)
    assert backbone.feature_dim == 1792

    x = torch.randn(2, 3, 224, 224)
    feats = backbone.extract_features(x)
    assert feats.shape == (2, 1792)

    out = backbone(x)
    assert out.shape == (2, 1792)
    assert backbone.get_num_parameters() > 0


def test_feature_extractor_4d_and_5d():
    backbone = EfficientNetB4Backbone(pretrained=False)
    extractor = FeatureExtractor(backbone)

    # 4D Image batch: (B, C, H, W)
    x_4d = torch.randn(2, 3, 224, 224)
    feats_4d = extractor(x_4d)
    assert feats_4d.shape == (2, 1792)

    # 5D Video sequence batch: (B, T, C, H, W)
    x_5d = torch.randn(2, 4, 3, 224, 224)
    feats_5d = extractor(x_5d)
    assert feats_5d.shape == (2, 4, 1792)


def test_freeze_and_unfreeze():
    backbone = EfficientNetB4Backbone(pretrained=False)
    assert backbone.get_trainable_parameters() > 0

    backbone.freeze()
    assert backbone.get_trainable_parameters() == 0

    backbone.unfreeze()
    assert backbone.get_trainable_parameters() == backbone.get_num_parameters()

    backbone.freeze_layers(until_stage=2)
    assert 0 < backbone.get_trainable_parameters() < backbone.get_num_parameters()


def test_modular_classifier_head():
    head = ModularClassifierHead(in_features=1792, num_classes=2, dropout=0.2, norm_layer="layernorm")
    feats = torch.randn(4, 1792)
    logits = head(feats)
    assert logits.shape == (4, 2)

    # Multiclass test
    head_multi = ModularClassifierHead(in_features=1792, num_classes=5)
    logits_multi = head_multi(feats)
    assert logits_multi.shape == (4, 5)


def test_efficientnet_model_modes():
    cfg = ModelConfig(backbone_name="efficientnet_b4", pretrained=False, num_classes=2)
    model = EfficientNetB4Model(config=cfg)

    # Standard inference/eval mode
    model.set_mode(ExecutionMode.INFERENCE)
    x_4d = torch.randn(2, 3, 224, 224)
    logits_4d = model(x_4d)
    assert logits_4d.shape == (2, 2)

    # 5D Sequence input
    x_5d = torch.randn(2, 4, 3, 224, 224)
    logits_5d = model(x_5d)
    assert logits_5d.shape == (2, 2)

    # Spatial feature extraction
    feats = model.extract_features(x_5d)
    assert feats.shape == (2, 4, 1792)

    # Temporal clip embedding extraction
    clip_emb = model.extract_clip_embedding(x_5d)
    assert clip_emb.shape == (2, 1792)


def test_weight_loader(tmp_path):
    cfg = ModelConfig(backbone_name="efficientnet_b4", pretrained=False)
    model = EfficientNetB4Model(config=cfg)
    chk_path = tmp_path / "model_weights.pt"
    torch.save(model.state_dict(), chk_path)

    new_model = EfficientNetB4Model(config=cfg)
    res = WeightLoader.load_weights(new_model, chk_path)
    assert res is not None


def test_model_registry_and_builder():
    assert "efficientnet_b4" in model_registry.list_registered()

    builder = VideoModelBuilder()
    cfg = ModelConfig(backbone_name="efficientnet_b4", pretrained=False)
    built_model = builder.build(cfg)
    assert isinstance(built_model, EfficientNetB4Model)


def test_mixed_precision_compatibility():
    model = EfficientNetB4Model(config=ModelConfig(pretrained=False))
    x = torch.randn(2, 3, 224, 224)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
        out = model(x)
    assert out.shape == (2, 2)
