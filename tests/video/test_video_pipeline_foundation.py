"""Executable Phase 1 Video Pipeline Foundation Smoke Test.

Verifies end-to-end video data flow without training:
1. Scans real MP4 videos from datasets/video/celebdfv2 or datasets/video/faceforensics
2. Decodes raw video frames using VideoDecoder (OpenCV)
3. Uniformly samples T=16 frames using FrameSampler
4. Detects and crops faces using FaceDetector & FaceCropper
5. Preprocesses, resizes (224x224), and applies ImageNet normalization
6. Asserts tensor shape (16, 3, 224, 224) per sample
7. Collates batch using video_collate_fn into 5D tensor (B, 16, 3, 224, 224)
8. Verifies finite-value checks (no NaN, no Inf)
9. Verifies CPU/GPU device transfers
10. Passes batch through EfficientNetB4Model forward pass to produce (B, 2) logits
"""

import os
from pathlib import Path
import pytest
import torch

from app.video.configs.dataset_config import DatasetConfig
from app.video.configs.model_config import ModelConfig
from app.video.datasets.dataloader import video_collate_fn
from app.video.datasets.index_builder import DatasetIndexBuilder
from app.video.datasets.video_dataset import VideoDataset
from app.video.models.efficientnet.model import EfficientNetB4Model, ExecutionMode


def find_real_sample_videos(num_samples: int = 2) -> list[dict]:
    """Find real video files from datasets/video."""
    builder = DatasetIndexBuilder()
    items = builder._build_index_items("celebdfv2", "train")
    real_files = [item for item in items if os.path.exists(item.sample_path) and not item.sample_path.startswith("synthetic")]

    if len(real_files) < num_samples:
        items_ff = builder._build_index_items("faceforensics", "train")
        real_files.extend([item for item in items_ff if os.path.exists(item.sample_path) and not item.sample_path.startswith("synthetic")])

    if len(real_files) < num_samples:
        pytest.skip(f"Not enough real video files found for smoke test (found {len(real_files)}, required {num_samples})")

    selected = real_files[:num_samples]
    return [
        {
            "filepath": item.sample_path,
            "label": item.label,
            "sample_id": f"smoke_test_{idx}",
        }
        for idx, item in enumerate(selected)
    ]


def test_video_pipeline_foundation_smoke_test():
    """Phase 1 Video Pipeline Foundation Smoke Test."""
    samples = find_real_sample_videos(num_samples=2)
    print("\n--- PHASE 1 VIDEO PIPELINE FOUNDATION SMOKE TEST ---")
    for s in samples:
        print(f"Sample Video Path: {s['filepath']} (Label: {s['label']})")

    # 1. Dataset Config & Instantiation
    ds_config = DatasetConfig(
        dataset_name="celeb_df_v2",
        sequence_length=16,
        target_resolution=(224, 224),
        crop_faces=True,
        sampling_strategy="uniform",
    )
    dataset = VideoDataset(config=ds_config, samples=samples)
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"

    # 2. Individual Sample Loading & Tensor Checks
    loaded_samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.tensor is not None, f"Sample {i} tensor is None"
        assert sample.tensor.ndim == 4, f"Expected 4D tensor (T, C, H, W), got {sample.tensor.shape}"
        assert sample.tensor.shape == (16, 3, 224, 224), f"Expected shape (16, 3, 224, 224), got {sample.tensor.shape}"

        # Finite checks
        assert not torch.isnan(sample.tensor).any(), f"Sample {i} tensor contains NaN"
        assert not torch.isinf(sample.tensor).any(), f"Sample {i} tensor contains Inf"

        print(f"Sample {i} loaded successfully: shape={sample.tensor.shape}, dtype={sample.tensor.dtype}")
        loaded_samples.append({"tensor": sample.tensor, "label": sample.label, "filepath": sample.filepath})

    # 3. Batch Collation
    batch_dict = video_collate_fn(loaded_samples)
    batch_tensor = batch_dict["tensor"]
    batch_labels = batch_dict["label"]

    assert batch_tensor.ndim == 5, f"Expected 5D batch tensor (B, T, C, H, W), got {batch_tensor.shape}"
    assert batch_tensor.shape == (2, 16, 3, 224, 224), f"Expected batch shape (2, 16, 3, 224, 224), got {batch_tensor.shape}"
    assert batch_labels.shape == (2,), f"Expected label shape (2,), got {batch_labels.shape}"
    assert not torch.isnan(batch_tensor).any(), "Batch tensor contains NaN"
    assert not torch.isinf(batch_tensor).any(), "Batch tensor contains Inf"

    print(f"Batch Collation Success: 5D Tensor Shape = {batch_tensor.shape}, Labels = {batch_labels.tolist()}")

    # 4. Device Transfers & Model Forward Pass
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")

    for device_str in devices_to_test:
        device = torch.device(device_str)
        print(f"Testing Model Forward Pass on device: {device_str.upper()}")

        model_config = ModelConfig(
            model_name="video_detector",
            backbone_name="efficientnet_b4",
            num_classes=2,
            sequence_length=16,
            pretrained=False,
        )
        model = EfficientNetB4Model(config=model_config).to(device)
        model.set_mode(ExecutionMode.INFERENCE)

        dev_batch = batch_tensor.to(device)
        with torch.no_grad():
            logits = model(dev_batch)

        assert logits.ndim == 2, f"Expected 2D logits (B, 2), got {logits.shape}"
        assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
        assert not torch.isnan(logits).any(), f"Logits contain NaN on {device_str}"
        assert not torch.isinf(logits).any(), f"Logits contain Inf on {device_str}"

        print(f"Forward Pass Success on {device_str.upper()}: Logits shape = {logits.shape}, values =\n{logits.cpu().numpy()}")

    print("--- PHASE 1 SMOKE TEST PASSED SUCCESSFULLY ---\n")
