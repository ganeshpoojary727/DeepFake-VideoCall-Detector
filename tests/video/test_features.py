"""Unit tests for feature extractor interfaces and registry."""

import pytest
import torch
from app.video.features.base_feature_extractor import BaseFeatureExtractor
from app.video.features.feature_registry import FeatureExtractorRegistry, feature_registry
from app.video.features.spatial_feature_extractor import SpatialFeatureExtractor
from app.video.features.temporal_feature_extractor import TemporalFeatureExtractor


class DummySpatialExtractor(SpatialFeatureExtractor):
    def extract_spatial_features(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[:2]
        return torch.zeros(b, t, self.output_dim)


class DummyTemporalExtractor(TemporalFeatureExtractor):
    def extract_temporal_features(self, spatial_features: torch.Tensor) -> torch.Tensor:
        b = spatial_features.shape[0]
        return torch.zeros(b, self.output_dim)


def test_spatial_feature_extractor():
    ext = DummySpatialExtractor(out_dim=1792)
    assert ext.output_dim == 1792
    frames = torch.zeros(2, 4, 3, 224, 224)
    out = ext(frames)
    assert out.shape == (2, 4, 1792)


def test_temporal_feature_extractor():
    ext = DummyTemporalExtractor(out_dim=512)
    assert ext.output_dim == 512
    spatial_feats = torch.zeros(2, 4, 1792)
    out = ext(spatial_feats)
    assert out.shape == (2, 512)


def test_feature_registry():
    reg = FeatureExtractorRegistry()
    reg.register("dummy_spatial", DummySpatialExtractor)
    cls = reg.get("dummy_spatial")
    assert cls == DummySpatialExtractor
    assert "dummy_spatial" in reg.list_registered()
