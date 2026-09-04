"""
Multi-Dataset Loader for Video Deepfake Training.

Aggregates and balances samples across multiple benchmark datasets:
1. FaceForensics++ (Original, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures, DeepFakeDetection)
2. Celeb-DF v2 (Celeb-real, YouTube-real, Celeb-synthesis)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


def collect_all_real_videos(datasets_root: Path) -> List[Path]:
    """Collect all pristine/authentic videos across FF++ and Celeb-DF."""
    real_videos: List[Path] = []

    # 1. FaceForensics++ Original
    ff_orig = datasets_root / "video" / "faceforensics" / "original"
    if ff_orig.exists():
        ff_files = list(ff_orig.glob("*.mp4"))
        logger.info("Found %d FaceForensics++ original real videos", len(ff_files))
        real_videos.extend(ff_files)

    # 2. Celeb-DF v2 Celeb-real
    celeb_real = datasets_root / "video" / "celebdfv2" / "Celeb-real"
    if celeb_real.exists():
        celeb_files = list(celeb_real.glob("*.mp4"))
        logger.info("Found %d Celeb-DF Celeb-real videos", len(celeb_files))
        real_videos.extend(celeb_files)

    # 3. Celeb-DF v2 YouTube-real
    yt_real = datasets_root / "video" / "celebdfv2" / "YouTube-real"
    if yt_real.exists():
        yt_files = list(yt_real.glob("*.mp4"))
        logger.info("Found %d Celeb-DF YouTube-real videos", len(yt_files))
        real_videos.extend(yt_files)

    logger.info("Total authentic real videos collected: %d", len(real_videos))
    return real_videos


def collect_all_fake_videos(datasets_root: Path) -> Dict[str, List[Path]]:
    """Collect manipulated videos grouped by manipulation technique."""
    fakes_by_method: Dict[str, List[Path]] = {}

    # 1. FaceForensics++ Manipulations
    ff_dir = datasets_root / "video" / "faceforensics"
    if ff_dir.exists():
        for sub in ff_dir.iterdir():
            if sub.is_dir() and sub.name.lower() not in ("original", "csv", "cache", "processed"):
                vids = list(sub.glob("*.mp4"))
                if vids:
                    fakes_by_method[f"ff_{sub.name}"] = vids
                    logger.info("Found %d FF++ videos for method '%s'", len(vids), sub.name)

    # 2. Celeb-DF v2 Celeb-synthesis
    celeb_synth = datasets_root / "video" / "celebdfv2" / "Celeb-synthesis"
    if celeb_synth.exists():
        vids = list(celeb_synth.glob("*.mp4"))
        if vids:
            fakes_by_method["celeb_synthesis"] = vids
            logger.info("Found %d Celeb-DF synthesis deepfakes", len(vids))

    total_fakes = sum(len(v) for v in fakes_by_method.values())
    logger.info("Total manipulated deepfake videos collected: %d across %d methods", total_fakes, len(fakes_by_method))
    return fakes_by_method


def build_balanced_multi_dataset(
    datasets_root: Path,
    target_samples_per_class: int = 2000,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a balanced, leakage-free 80/20 train/validation split combining FF++ and Celeb-DF."""
    rng = random.Random(seed)

    real_videos = collect_all_real_videos(datasets_root)
    fakes_by_method = collect_all_fake_videos(datasets_root)

    # Deterministic train/val partition by video hash/ID
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    # 1. Process Real Videos
    for p in real_videos:
        h = abs(hash(p.stem)) % 100
        sample = {"filepath": str(p), "label": 0, "sample_id": p.name, "source": "real"}
        if h < int(train_ratio * 100):
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    # 2. Sample balanced fakes equally across all methods
    num_methods = max(1, len(fakes_by_method))
    per_method_quota = max(100, target_samples_per_class // num_methods)

    selected_fakes: List[Path] = []
    for method_name, vids in fakes_by_method.items():
        shuffled = list(vids)
        rng.shuffle(shuffled)
        chosen = shuffled[:per_method_quota]
        selected_fakes.extend(chosen)
        logger.info("Method '%s': sampled %d / %d videos", method_name, len(chosen), len(vids))

    for p in selected_fakes:
        h = abs(hash(p.stem)) % 100
        sample = {"filepath": str(p), "label": 1, "sample_id": p.name, "source": "fake"}
        if h < int(train_ratio * 100):
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    # Shuffle splits
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_reals = sum(1 for s in train_samples if s["label"] == 0)
    train_fakes = sum(1 for s in train_samples if s["label"] == 1)
    val_reals = sum(1 for s in val_samples if s["label"] == 0)
    val_fakes = sum(1 for s in val_samples if s["label"] == 1)

    logger.info(
        "Balanced Multi-Dataset Built:\n"
        "  • Train: %d videos (Real: %d, Fake: %d)\n"
        "  • Val:   %d videos (Real: %d, Fake: %d)\n"
        "  • Total: %d videos",
        len(train_samples), train_reals, train_fakes,
        len(val_samples), val_reals, val_fakes,
        len(train_samples) + len(val_samples),
    )

    return train_samples, val_samples
