"""Unit tests for video base models, backbones, attention, classifiers, loader, and factory."""

import os
import tempfile
import pytest
import torch

from app.video.configs.model_config import ModelConfig
from app.video.exceptions.video_exceptions import ModelError
from app.video.models.attention.base_attention import (
    AttentionRegistry,
    BaseTemporalAttention,
    DummyTemporalAttention,
    attention_registry,
)
from app.video.models.backbones.base_backbone import (
    BackboneRegistry,
    BaseBackbone,
    DummyBackbone,
    backbone_registry,
)
from app.video.models.base_video_model import BaseVideoModel
from app.video.models.classifiers.base_classifier import (
    BaseClassifier,
    ClassifierRegistry,
    LinearClassifier,
    classifier_registry,
)
from app.video.models.model_loader import VideoModelLoader
from app.video.models.model_registry import VideoModelRegistry, model_registry
from app.video.models.video_factory import ModularVideoModel, VideoFactory


def test_dummy_backbone():
    bb = DummyBackbone(in_channels=3, feature_dim=1792)
    assert bb.in_channels == 3
    assert bb.feature_dim == 1792
    x = torch.zeros(4, 3, 224, 224)
    feat = bb(x)
    assert feat.shape == (4, 1792)


def test_dummy_backbone_properties():
    bb = DummyBackbone(in_channels=1, feature_dim=512)
    assert bb.in_channels == 1
    assert bb.feature_dim == 512


def test_dummy_attention():
    att = DummyTemporalAttention(feature_dim=1792, out_dim=512)
    assert att.feature_dim == 1792
    assert att.output_dim == 512
    seq_feats = torch.zeros(2, 8, 1792)
    out = att(seq_feats)
    assert out.shape == (2, 512)


def test_dummy_attention_properties():
    att = DummyTemporalAttention(feature_dim=512, out_dim=256)
    assert att.feature_dim == 512
    assert att.output_dim == 256


def test_linear_classifier():
    clf = LinearClassifier(in_features=512, num_classes=2)
    assert clf.in_features == 512
    assert clf.num_classes == 2
    feats = torch.zeros(2, 512)
    logits = clf(feats)
    assert logits.shape == (2, 2)


def test_linear_classifier_properties():
    clf = LinearClassifier(in_features=256, num_classes=5)
    assert clf.in_features == 256
    assert clf.num_classes == 5


def test_modular_video_model():
    bb = DummyBackbone()
    att = DummyTemporalAttention()
    clf = LinearClassifier()
    model = ModularVideoModel(backbone=bb, attention=att, classifier=clf)

    video = torch.zeros(2, 4, 3, 224, 224)
    logits = model(video)
    assert logits.shape == (2, 2)
    total, trainable = model.get_num_parameters()
    assert total >= 0


def test_modular_video_model_extract_features():
    bb = DummyBackbone()
    att = DummyTemporalAttention()
    clf = LinearClassifier()
    model = ModularVideoModel(backbone=bb, attention=att, classifier=clf)
    video = torch.zeros(2, 4, 3, 224, 224)
    out = model.extract_features(video)
    assert out.shape == (2, 2)


def test_video_factory_create():
    cfg = ModelConfig(backbone_name="dummy_backbone", num_classes=2)
    model = VideoFactory.create_model(cfg)
    assert isinstance(model, BaseVideoModel)
    assert model.num_classes == 2


def test_model_registry_list():
    models = model_registry.list_registered()
    assert "modular_video_model" in models


def test_model_loader_save_and_load():
    cfg = ModelConfig()
    model = VideoFactory.create_model(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model.pt")
        VideoModelLoader.save_model(model, ckpt_path, extra_meta={"version": "1.0"})
        assert os.path.exists(ckpt_path)

        new_model = VideoFactory.create_model(cfg)
        payload = VideoModelLoader.load_model(new_model, ckpt_path)
        assert payload["extra_meta"]["version"] == "1.0"


def test_model_loader_missing_file():
    cfg = ModelConfig()
    model = VideoFactory.create_model(cfg)
    with pytest.raises(ModelError):
        VideoModelLoader.load_model(model, "non_existent_file.pt")
