"""Test PyTorch DataLoader multi-processing speed on Windows."""

import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video.datasets.video_dataset import VideoDataset
from app.video.configs.dataset_config import DatasetConfig
from app.video.datasets.dataloader import video_collate_fn

def main():
    original_dir = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"
    deepfakes_dir = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "Deepfakes"

    real_videos = sorted(list(original_dir.glob("*.mp4")))[:20]
    fake_videos = sorted(list(deepfakes_dir.glob("*.mp4")))[:20]

    samples = []
    for p in real_videos:
        samples.append({"filepath": str(p), "label": 0, "sample_id": p.name})
    for p in fake_videos:
        samples.append({"filepath": str(p), "label": 1, "sample_id": p.name})

    ds = VideoDataset(config=DatasetConfig(sequence_length=16, crop_faces=True), samples=samples)

    for nw in [0, 2, 4]:
        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            num_workers=nw,
            collate_fn=video_collate_fn,
            persistent_workers=(nw > 0),
        )
        t0 = time.perf_counter()
        count = 0
        for batch in loader:
            count += len(batch["label"])
        t1 = time.perf_counter()
        print(f"num_workers={nw}: {t1 - t0:.2f}s for {count} samples ({(t1 - t0)/count:.2f}s per sample)")

if __name__ == "__main__":
    main()
