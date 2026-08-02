"""Composite video augmentation pipeline container module."""

from __future__ import annotations

from typing import List, Optional
import torch

from app.video.augmentation.base_augmentation import BaseVideoAugmentation
from app.video.augmentation.blur_noise import Blur, GaussianBlur, GaussianNoise, Noise
from app.video.augmentation.color_adjust import Brightness, ColorJitter, Contrast
from app.video.augmentation.compression import JPEG, JPEGCompression, RandomCompression
from app.video.augmentation.spatial_aug import HorizontalFlip, RandomCrop, RandomRotation, Rotation
from app.video.augmentation.temporal_aug import FrameDropout, FrameSkip, TemporalDrop, TemporalJitter
from app.video.configs.augmentation_config import AugmentationConfig


class VideoAugmentationPipeline:
    """Composes multiple spatial and temporal video augmentations into a sequence chain."""

    def __init__(
        self,
        transforms: Optional[List[BaseVideoAugmentation]] = None,
        config: Optional[AugmentationConfig] = None,
    ) -> None:
        if transforms is not None:
            self._transforms = transforms
        elif config is not None:
            self._transforms = self._build_from_config(config)
        else:
            self._transforms = []

    def _build_from_config(self, cfg: AugmentationConfig) -> List[BaseVideoAugmentation]:
        if not cfg.enable_augmentation:
            return []

        aug_list: List[BaseVideoAugmentation] = []

        if cfg.enable_horizontal_flip:
            aug_list.append(HorizontalFlip(p=cfg.horizontal_flip_prob))

        if cfg.enable_random_crop:
            aug_list.append(RandomCrop(crop_size=cfg.crop_size, p=cfg.crop_prob))

        if cfg.enable_rotation:
            aug_list.append(RandomRotation(max_degrees=cfg.max_rotation_degrees, p=cfg.rotation_prob))

        if cfg.enable_color_jitter:
            aug_list.append(
                ColorJitter(
                    brightness=cfg.brightness_factor,
                    contrast=cfg.contrast_factor,
                    p=cfg.color_jitter_prob,
                )
            )

        if cfg.enable_blur:
            aug_list.append(GaussianBlur(kernel_size=cfg.blur_kernel_size, p=cfg.blur_prob))

        if cfg.enable_jpeg_compression:
            aug_list.append(RandomCompression(quality_range=cfg.jpeg_quality_range, p=cfg.jpeg_prob))

        if cfg.enable_noise:
            aug_list.append(GaussianNoise(std=cfg.noise_std, p=cfg.noise_prob))

        if cfg.enable_frame_dropout:
            aug_list.append(FrameDropout(drop_prob=cfg.frame_drop_prob, p=cfg.temporal_drop_prob))

        if cfg.enable_temporal_jitter:
            aug_list.append(TemporalJitter(max_shift=cfg.temporal_jitter_max_shift, p=cfg.temporal_jitter_prob))

        return aug_list

    @property
    def transforms(self) -> List[BaseVideoAugmentation]:
        """Get registered transform list."""
        return self._transforms

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply all transforms sequentially onto video tensor [T, C, H, W]."""
        for t in self._transforms:
            video = t(video)
        return video
