"""Audit FaceForensics++ dataset structure and verify train/val source separation."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FFPP_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics"

def extract_source_ids(filepath: Path) -> Tuple[str, ...]:
    """Extract source video ID(s) from filename.
    
    FaceForensics++ naming conventions:
    - Original real: '000.mp4' -> source ID ('000',)
    - Manipulated fake: '000_003.mp4' -> source IDs ('000', '003')
    - DeepFakeDetection real: '01__kitchen...mp4' or similar
    - DeepFakeDetection fake: '01_02__kitchen...mp4'
    """
    stem = filepath.stem
    if "__" in stem:
        # DeepFakeDetection format
        id_part = stem.split("__")[0]
        ids = tuple(id_part.split("_"))
    elif "_" in stem:
        # Standard FF++ manipulation format e.g. 000_003
        ids = tuple(stem.split("_"))
    else:
        # Original format e.g. 000
        ids = (stem,)
    return ids

def audit_dataset():
    print("==================================================")
    print("  FACEFORENSICS++ DATASET AUDIT & LEAKAGE CHECK")
    print("==================================================")

    categories = [d.name for d in FFPP_DIR.iterdir() if d.is_dir() and d.name != "csv"]
    print(f"Categories found in {FFPP_DIR}: {categories}")

    category_samples: Dict[str, List[Path]] = {}
    for cat in categories:
        cat_dir = FFPP_DIR / cat
        files = sorted(list(cat_dir.glob("*.mp4")))
        category_samples[cat] = files
        print(f"Category '{cat}': {len(files)} videos")

    # Group videos by primary source ID
    # In standard FF++, IDs are 000 to 999.
    # Standard split in literature:
    # Train: 720 videos (IDs 000-719)
    # Val:   140 videos (IDs 720-859)
    # Test:  140 videos (IDs 860-999)
    #
    # Let's check how many videos fall into IDs < 800 vs >= 800 or 80/20 split!
    
    all_real = category_samples.get("original", [])
    all_fakes = []
    for cat, files in category_samples.items():
        if cat.lower() in ("deepfakes", "face2face", "faceswap", "faceshifter", "neuraltextures", "deepfakedetection"):
            all_fakes.extend(files)

    print(f"\nTotal Real videos (original): {len(all_real)}")
    print(f"Total Fake videos (all fake categories): {len(all_fakes)}")
    print(f"Grand Total videos: {len(all_real) + len(all_fakes)}")

    # Deterministic split by source ID:
    # Train IDs: 000..799 (80% of 1000 base IDs)
    # Val IDs:   800..999 (20% of 1000 base IDs)
    # Any video whose source IDs overlap with train IDs goes to train.
    # Any video whose source IDs overlap with val IDs goes to val.
    # NO video can have source IDs crossing the boundary (e.g. 050_850).
    
    train_real, val_real = [], []
    train_fake, val_fake = [], []
    boundary_cross_fakes = []

    for p in all_real:
        ids = extract_source_ids(p)
        primary = int(ids[0])
        if primary < 800:
            train_real.append(p)
        else:
            val_real.append(p)

    for p in all_fakes:
        ids = extract_source_ids(p)
        try:
            num_ids = [int(x) for x in ids if x.isdigit()]
        except ValueError:
            num_ids = []

        if not num_ids:
            # Fallback for non-numeric names if any
            h = hash(p.name) % 10
            if h < 8:
                train_fake.append(p)
            else:
                val_fake.append(p)
            continue

        in_train = any(i < 800 for i in num_ids)
        in_val = any(i >= 800 for i in num_ids)

        if in_train and in_val:
            # Crosses boundary! Put in train to be conservative, or exclude from val
            boundary_cross_fakes.append(p)
            train_fake.append(p)
        elif in_train:
            train_fake.append(p)
        else:
            val_fake.append(p)

    # Verify Leakage:
    train_sources: Set[str] = set()
    for p in train_real + train_fake:
        train_sources.update(extract_source_ids(p))

    val_sources: Set[str] = set()
    for p in val_real + val_fake:
        val_sources.update(extract_source_ids(p))

    overlap_sources = train_sources.intersection(val_sources)

    print("\n--- DATASET AUDIT SUMMARY ---")
    print(f"Train Real:  {len(train_real)}")
    print(f"Train Fake:  {len(train_fake)}")
    print(f"Train Total: {len(train_real) + len(train_fake)}")
    print(f"Val Real:    {len(val_real)}")
    print(f"Val Fake:    {len(val_fake)}")
    print(f"Val Total:   {len(val_real) + len(val_fake)}")
    print(f"Boundary Crossing Fakes: {len(boundary_cross_fakes)}")
    print(f"Source Overlap Count: {len(overlap_sources)}")
    if overlap_sources:
        print(f"OVERLAP SOURCES: {overlap_sources}")
    else:
        print("DATA LEAKAGE CHECK PASSED: 0 source video overlap between train and validation!")

if __name__ == "__main__":
    audit_dataset()
