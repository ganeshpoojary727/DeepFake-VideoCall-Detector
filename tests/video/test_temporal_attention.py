"""Exhaustive unit tests for production Temporal Attention Subsystem."""

import pytest
import torch

from app.video.configs import ModelConfig
from app.video.models import EfficientNetB4Model, ExecutionMode
from app.video.models.attention import (
    TemporalEncoder,
    TemporalFeatureExtractor,
    TemporalMultiHeadAttention,
    TemporalPooling,
    TemporalPositionalEncoding,
    TemporalTransformerBlock,
    attention_registry,
)


def test_positional_encoding():
    pos_enc = TemporalPositionalEncoding(max_len=64, feature_dim=1792)
    x = torch.randn(2, 16, 1792)
    out = pos_enc(x)
    assert out.shape == (2, 16, 1792)

    # Test variable sequence length exceeding max_len (interpolation)
    x_long = torch.randn(2, 80, 1792)
    out_long = pos_enc(x_long)
    assert out_long.shape == (2, 80, 1792)


def test_multihead_attention():
    mha = TemporalMultiHeadAttention(feature_dim=1792, num_heads=8, attn_dropout=0.1)
    x = torch.randn(2, 16, 1792)
    out = mha(x)
    assert out.shape == (2, 16, 1792)


def test_transformer_block():
    block = TemporalTransformerBlock(feature_dim=1792, num_heads=8, ffn_dim=2048)
    x = torch.randn(2, 16, 1792)
    out = block(x)
    assert out.shape == (2, 16, 1792)


def test_temporal_pooling_modes():
    x = torch.randn(2, 16, 1792)

    for mode in ["mean", "max", "attention"]:
        pooler = TemporalPooling(feature_dim=1792, pooling_type=mode)
        out = pooler(x)
        assert out.shape == (2, 1792)

    cls_pooler = TemporalPooling(feature_dim=1792, pooling_type="cls")
    x_cls = cls_pooler.prepend_cls_token(x)
    assert x_cls.shape == (2, 17, 1792)
    out_cls = cls_pooler(x_cls)
    assert out_cls.shape == (2, 1792)


def test_temporal_encoder_end_to_end():
    cfg = ModelConfig(feature_dim=1792, num_heads=8, num_layers=2, pooling_type="attention")
    encoder = TemporalEncoder(feature_dim=1792, out_dim=1792, config=cfg)
    x = torch.randn(2, 16, 1792)
    out = encoder(x)
    assert out.shape == (2, 1792)


def test_temporal_feature_extractor():
    cfg = ModelConfig(feature_dim=1792, num_heads=8, num_layers=1)
    encoder = TemporalEncoder(feature_dim=1792, out_dim=1792, config=cfg)
    extractor = TemporalFeatureExtractor(encoder)

    x = torch.randn(2, 8, 1792)
    out = extractor.extract_clip_embedding(x)
    assert out.shape == (2, 1792)


def test_gradient_flow_and_mixed_precision():
    cfg = ModelConfig(feature_dim=1792, num_heads=8, num_layers=2)
    encoder = TemporalEncoder(feature_dim=1792, out_dim=1792, config=cfg)
    x = torch.randn(2, 16, 1792, requires_grad=True)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
        out = encoder(x)
        loss = out.sum()

    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 16, 1792)


def test_attention_registry():
    assert "temporal_transformer" in attention_registry.list_registered()
    assert "temporal_encoder" in attention_registry.list_registered()


def test_efficientnet_model_temporal_integration():
    cfg = ModelConfig(backbone_name="efficientnet_b4", pretrained=False, num_classes=2)
    model = EfficientNetB4Model(config=cfg)

    # 5D Video input: (B, T, C, H, W)
    x_video = torch.randn(2, 4, 3, 224, 224)
    logits = model(x_video)
    assert logits.shape == (2, 2)

    # Clip embedding extraction
    model.set_mode(ExecutionMode.FEATURE_EXTRACTION)
    clip_emb = model(x_video)
    assert clip_emb.shape == (2, 1792)
