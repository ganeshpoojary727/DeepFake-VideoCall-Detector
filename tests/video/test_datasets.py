"""Unit tests for video dataset classes and sample structures."""

import pytest
import torch
from app.video.configs.dataset_config import DatasetConfig
from app.video.datasets.celeb_df_dataset import CelebDFDataset
from app.video.datasets.dataset_factory import DatasetFactory
from app.video.datasets.deepforensics_dataset import DeepForensicsDataset
from app.video.datasets.faceforensics_dataset import FaceForensicsDataset
from app.video.datasets.metadata import (
    DatasetMetadata,
    FaceMetadata,
    SampleMetadata,
    VideoMetadata,
)
from app.video.datasets.mixed_dataset import MixedVideoDataset
from app.video.datasets.video_dataset import VideoDataset
from app.video.datasets.video_sample import VideoSample
from app.video.exceptions.video_exceptions import DatasetError


def test_video_metadata_structures():
    v_meta = VideoMetadata(filepath="video.mp4", duration_sec=5.0, total_frames=150)
    assert v_meta.filepath == "video.mp4"
    assert v_meta.total_frames == 150

    f_meta = FaceMetadata(bbox=(10, 10, 50, 50), confidence=0.95)
    assert f_meta.bbox == (10, 10, 50, 50)

    s_meta = SampleMetadata(
        sample_id="s1", label=1, dataset_name="ffpp", video_meta=v_meta, face_meta=f_meta
    )
    assert s_meta.label == 1

    d_meta = DatasetMetadata(
        dataset_name="ffpp", num_samples=100, num_reals=50, num_fakes=50
    )
    assert d_meta.num_samples == 100


def test_video_sample():
    tensor = torch.zeros(16, 3, 224, 224)
    sample = VideoSample(
        tensor=tensor, label=0, filepath="sample.mp4", sample_id="id1"
    )
    assert sample.label == 0
    assert sample.tensor.shape == (16, 3, 224, 224)


def test_video_sample_invalid_tensor():
    with pytest.raises(TypeError):
        VideoSample(
            tensor="not_a_tensor", label=0, filepath="sample.mp4", sample_id="id1"
        )


def test_base_video_dataset():
    cfg = DatasetConfig(sequence_length=4)
    samples = [{"filepath": "v1.mp4", "label": 0, "sample_id": "s1"}]
    ds = VideoDataset(config=cfg, samples=samples)
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample, VideoSample)
    assert sample.tensor.shape == (4, 3, 224, 224)


def test_faceforensics_dataset():
    cfg = DatasetConfig(sequence_length=4)
    ds = FaceForensicsDataset(config=cfg, samples=[{"label": 1}])
    assert len(ds) == 1
    assert ds.compression_level == "c23"


def test_celeb_df_dataset():
    cfg = DatasetConfig(sequence_length=4)
    ds = CelebDFDataset(config=cfg, samples=[{"label": 0}])
    assert len(ds) == 1


def test_deepforensics_dataset():
    cfg = DatasetConfig(sequence_length=4)
    ds = DeepForensicsDataset(config=cfg, samples=[{"label": 1}])
    assert len(ds) == 1


def test_mixed_video_dataset():
    ds1 = VideoDataset(samples=[{"label": 0}])
    ds2 = VideoDataset(samples=[{"label": 1}])
    mixed = MixedVideoDataset(datasets=[ds1, ds2])
    assert len(mixed) == 2
    assert len(mixed.datasets) == 2


def test_dataset_factory_create():
    cfg = DatasetConfig(dataset_name="faceforensics_pp", sequence_length=4)
    ds = DatasetFactory.create(config=cfg, samples=[{"label": 0}])
    assert isinstance(ds, FaceForensicsDataset)


def test_dataset_factory_invalid():
    cfg = DatasetConfig(dataset_name="non_existent_dataset")
    with pytest.raises(DatasetError):
        DatasetFactory.create(config=cfg)
