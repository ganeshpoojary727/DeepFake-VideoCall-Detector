"""Base dataset interface specification for Video Deepfake Forensics."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class BaseVideoDataset(ABC, Dataset):
    """Abstract base class for video PyTorch datasets (e.g. FaceForensics++, Celeb-DF v2).

    Standardizes sequence length (16 or 32 frames), resolution (224x224),
    and binary class labels (0 = REAL, 1 = FAKE).
    """

    def __init__(
        self,
        sequence_length: int = 16,
        target_resolution: Tuple[int, int] = (224, 224),
        samples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.target_resolution = target_resolution
        self.samples: List[Dict[str, Any]] = samples or []

    @abstractmethod
    def __len__(self) -> int:
        """Get total sample count in dataset split."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Fetch video sample at specified dataset index."""
        pass

    def get_label_distribution(self) -> Dict[int, int]:
        """Calculate sample distribution count per target class label (0=real, 1=fake)."""
        counts = {0: 0, 1: 0}
        for item in self.samples:
            label = item.get("label", 0)
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def parse_manifest_file(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Parse video dataset manifest files (JSON, CSV, or whitespace-delimited TXT).

        Supports:
        - JSON manifests: list of objects with "filepath" (or "video_path") and "label".
        - CSV manifests: columns `filepath,label` or `video,label,split`.
        - TXT manifests: e.g. Celeb-DF `List_of_testing_videos.txt` (`<label> <video_path>`).

        Returns:
            List[Dict[str, Any]]: Parsed record dictionaries.
        """
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        records: List[Dict[str, Any]] = []

        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            records.append(item)
                elif isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, dict):
                            records.append({"sample_id": key, **val})
                        elif isinstance(val, (int, str)):
                            records.append({"filepath": key, "label": int(val)})

        elif path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    label_val = row.get("label", row.get("target", "0"))
                    label = 1 if str(label_val).lower() in ("1", "fake", "spoof", "manipulated") else 0
                    fp = row.get("filepath", row.get("video_path", row.get("path", "")))
                    records.append({
                        **row,
                        "filepath": fp,
                        "label": int(label),
                        "sample_id": row.get("sample_id", Path(fp).stem if fp else ""),
                    })

        else:
            # Plain text / Celeb-DF format
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        # Celeb-DF testing format: "1 Celeb-synthesis/id0_id1_0000.mp4" or "0 YouTube-real/0000.mp4"
                        # Or format: "filepath 1"
                        if parts[0] in ("0", "1"):
                            label = int(parts[0])
                            filepath = parts[1]
                        elif parts[-1] in ("0", "1", "fake", "real", "bonafide", "spoof"):
                            label = 1 if parts[-1] in ("1", "fake", "spoof") else 0
                            filepath = parts[0]
                        else:
                            label = 0
                            filepath = parts[0]

                        records.append({
                            "filepath": filepath,
                            "label": label,
                            "sample_id": Path(filepath).stem,
                        })

        return records


# Backward compatibility alias
BaseDataset = BaseVideoDataset
