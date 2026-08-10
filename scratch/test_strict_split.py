"""Test strict zero-leakage FaceForensics++ dataset split."""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FFPP_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics"

def extract_source_ids(filepath: Path) -> Tuple[str, ...]:
    stem = filepath.stem
    if "__" in stem:
        id_part = stem.split("__")[0]
        ids = tuple(id_part.split("_"))
    elif "_" in stem:
        ids = tuple(stem.split("_"))
    else:
        ids = (stem,)
    return ids

def audit_strict_split():
    print("==================================================")
    print("  STRICT ZERO-LEAKAGE FF++ SPLIT CHECK")
    print("==================================================")

    categories = [d.name for d in FFPP_DIR.iterdir() if d.is_dir() and d.name != "csv"]
    category_samples: Dict[str, List[Path]] = {}
    for cat in categories:
        cat_dir = FFPP_DIR / cat
        files = sorted(list(cat_dir.glob("*.mp4")))
        category_samples[cat] = files

    all_real = category_samples.get("original", [])
    all_fakes = []
    for cat, files in category_samples.items():
        if cat.lower() in ("deepfakes", "face2face", "faceswap", "faceshifter", "neuraltextures", "deepfakedetection"):
            all_fakes.extend(files)

    train_real, val_real = [], []
    train_fake, val_fake = [], []
    excluded_cross_boundary = []

    # Rule: IDs 000..799 belong to train subject universe.
    # IDs 800..999 belong to val subject universe.
    # A fake video X_Y belongs to train ONLY if both X < 800 AND Y < 800.
    # A fake video X_Y belongs to val ONLY if both X >= 800 AND Y >= 800.
    # If X < 800 and Y >= 800, it is cross-boundary and excluded to prevent leakage.

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
            continue

        all_in_train = all(i < 800 for i in num_ids)
        all_in_val = all(i >= 800 for i in num_ids)

        if all_in_train:
            train_fake.append(p)
        elif all_in_val:
            val_fake.append(p)
        else:
            excluded_cross_boundary.append(p)

    train_sources: Set[str] = set()
    for p in train_real + train_fake:
        train_sources.update(extract_source_ids(p))

    val_sources: Set[str] = set()
    for p in val_real + val_fake:
        val_sources.update(extract_source_ids(p))

    overlap_sources = train_sources.intersection(val_sources)

    print("\n--- STRICT SPLIT RESULTS ---")
    print(f"Train Real:  {len(train_real)}")
    print(f"Train Fake:  {len(train_fake)}")
    print(f"Train Total: {len(train_real) + len(train_fake)}")
    print(f"Val Real:    {len(val_real)}")
    print(f"Val Fake:    {len(val_fake)}")
    print(f"Val Total:   {len(val_real) + len(val_fake)}")
    print(f"Excluded Cross-Boundary Videos: {len(excluded_cross_boundary)}")
    print(f"Source Overlap Count: {len(overlap_sources)}")

    if len(overlap_sources) == 0:
        print("\nSUCCESS: 0 Source Overlap! Strict train/val isolation verified.")
    else:
        print(f"\nWARNING: Source Overlap still present: {overlap_sources}")

if __name__ == "__main__":
    audit_strict_split()
