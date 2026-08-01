"""Unit tests for video constants."""

from app.video.constants import (
    BACKBONE_EFFICIENTNET_B4,
    DATASET_CELEB_DF,
    DATASET_DEEPFORENSICS,
    DATASET_FFPP,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    EFFICIENTNET_B4_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_DATASETS,
)


def test_spatial_constants():
    assert DEFAULT_FRAME_HEIGHT == 224
    assert DEFAULT_FRAME_WIDTH == 224
    assert EFFICIENTNET_B4_INPUT_SIZE == 380


def test_dataset_constants():
    assert DATASET_FFPP == "faceforensics_pp"
    assert DATASET_CELEB_DF == "celeb_df_v2"
    assert DATASET_DEEPFORENSICS == "deepforensics"
    assert len(SUPPORTED_DATASETS) == 3


def test_norm_constants():
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
