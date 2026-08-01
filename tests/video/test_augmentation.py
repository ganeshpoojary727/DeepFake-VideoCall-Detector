"""Unit tests for spatial and temporal video data augmentations."""

import pytest
import torch
from app.video.augmentation.augmentation_pipeline import VideoAugmentationPipeline
from app.video.augmentation.blur_noise import Blur, GaussianNoise, Noise
from app.video.augmentation.color_adjust import Brightness, ColorJitter, Contrast
from app.video.augmentation.compression import JPEG, RandomCompression
from app.video.augmentation.spatial_aug import HorizontalFlip, RandomCrop
from app.video.augmentation.temporal_aug import FrameSkip, TemporalDrop
from app.video.exceptions.video_exceptions import AugmentationError


def test_base_augmentation_invalid_prob():
    with pytest.raises(AugmentationError):
        JPEG(p=-0.5)


def test_jpeg_augmentation(dummy_video_tensor):
    aug = JPEG(quality=80, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_random_compression_augmentation(dummy_video_tensor):
    aug = RandomCompression(quality_range=(50, 90), p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_blur_augmentation(dummy_video_tensor):
    aug = Blur(kernel_size=3, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_gaussian_noise_augmentation(dummy_video_tensor):
    aug = GaussianNoise(std=0.05, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_noise_alias_augmentation(dummy_video_tensor):
    aug = Noise(std=0.05, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_brightness_augmentation(dummy_video_tensor):
    aug = Brightness(factor=0.2, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_contrast_augmentation(dummy_video_tensor):
    aug = Contrast(factor=0.2, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_color_jitter_augmentation(dummy_video_tensor):
    aug = ColorJitter(brightness=0.2, contrast=0.2, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_horizontal_flip_augmentation(dummy_video_tensor):
    aug = HorizontalFlip(p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_random_crop_augmentation(dummy_video_tensor):
    aug = RandomCrop(crop_size=(100, 100), p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == (16, 3, 100, 100)


def test_temporal_drop_augmentation(dummy_video_tensor):
    aug = TemporalDrop(drop_prob=0.2, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_frame_skip_augmentation(dummy_video_tensor):
    aug = FrameSkip(stride=2, p=1.0)
    res = aug(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape


def test_augmentation_pipeline(dummy_video_tensor):
    pipeline = VideoAugmentationPipeline(
        transforms=[
            HorizontalFlip(p=1.0),
            Brightness(factor=0.1, p=1.0),
        ]
    )
    res = pipeline(dummy_video_tensor)
    assert res.shape == dummy_video_tensor.shape
    assert len(pipeline.transforms) == 2
