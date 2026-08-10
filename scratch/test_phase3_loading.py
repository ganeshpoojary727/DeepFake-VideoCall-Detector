"""Test FaceForensics++ dataset loading speed and video counts."""

import sys
from pathlib import Path
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video.datasets.video_dataset import VideoDataset
from app.video.configs.dataset_config import DatasetConfig

def main():
    original_dir = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"
    deepfakes_dir = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "Deepfakes"

    real_videos = sorted(list(original_dir.glob("*.mp4")))
    fake_videos = sorted(list(deepfakes_dir.glob("*.mp4")))

    print(f"Total real videos in original: {len(real_videos)}")
    print(f"Total fake videos in Deepfakes: {len(fake_videos)}")

    # Select 100 real (000.mp4 to 099.mp4) and 100 fake (first 100 in Deepfakes)
    selected_real = real_videos[:100]
    selected_fake = fake_videos[:100]

    # Split 80/20 by ID prefix to prevent video subject leakage
    # Real IDs: 000..079 for train (80), 080..099 for val (20)
    # Fake videos matching these IDs
    train_samples = []
    val_samples = []

    for p in selected_real:
        vid_id = int(p.stem)
        sample = {"filepath": str(p), "label": 0, "sample_id": p.name}
        if vid_id < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    for p in selected_fake:
        # e.g., "000_003.mp4" -> primary ID is 000
        parts = p.stem.split("_")
        primary_id = int(parts[0])
        sample = {"filepath": str(p), "label": 1, "sample_id": p.name}
        if primary_id < 80:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    print(f"Train samples count: {len(train_samples)} (Real: {sum(1 for s in train_samples if s['label']==0)}, Fake: {sum(1 for s in train_samples if s['label']==1)})")
    print(f"Val samples count: {len(val_samples)} (Real: {sum(1 for s in val_samples if s['label']==0)}, Fake: {sum(1 for s in val_samples if s['label']==1)})")

    # Measure speed on 5 samples
    ds = VideoDataset(config=DatasetConfig(sequence_length=16, crop_faces=True), samples=train_samples[:5])
    t0 = time.perf_counter()
    for i in range(len(ds)):
        sample = ds[i]
        print(f"Sample {i}: shape={sample.tensor.shape}, label={sample.label}")
    t1 = time.perf_counter()
    print(f"Time for 5 video samples with face crop: {t1 - t0:.2f} seconds ({(t1 - t0)/5:.2f}s per video)")

if __name__ == "__main__":
    main()
