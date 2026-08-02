"""Prediction history recording, query, and export service."""

from __future__ import annotations

import csv

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Union

from app.core.interfaces import DetectionLabel, Modality, PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Dataclass representing a stored prediction record."""

    id: str
    timestamp: float
    confidence: float
    label: str
    modality: str
    latency_ms: float = 0.0
    model_version: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class HistoryService:
    """Stores inference prediction records in memory with query and export capabilities."""

    def __init__(self, max_records: int = 10000) -> None:
        self.max_records = max_records
        self._records: List[PredictionRecord] = []
        self._lock = threading.Lock()
        self._record_counter: int = 0

    def add_prediction(
        self,
        prediction: Union[PredictionResult, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionRecord:
        """Record a new detection prediction item.

        Args:
            prediction: PredictionResult object or raw result dictionary.
            metadata: Optional additional metadata dict.

        Returns:
            PredictionRecord: Added record instance.
        """
        now = time.time()
        with self._lock:
            self._record_counter += 1
            rec_id = f"rec_{self._record_counter:08d}"

            if isinstance(prediction, PredictionResult):
                label_str = prediction.label.value if hasattr(prediction.label, "value") else str(prediction.label)
                mod_str = prediction.modality.value if hasattr(prediction.modality, "value") else str(prediction.modality)
                conf = prediction.confidence
                lat = prediction.latency_ms
                ver = prediction.model_version
            else:
                label_str = str(prediction.get("label_name", prediction.get("label", "REAL")))
                mod_str = str(prediction.get("modality", "fused"))
                conf = float(prediction.get("confidence", prediction.get("fake_probability", 0.0)))
                lat = float(prediction.get("latency_ms", 0.0))
                ver = str(prediction.get("model_version", "unknown"))

            rec = PredictionRecord(
                id=rec_id,
                timestamp=now,
                confidence=conf,
                label=label_str.upper(),
                modality=mod_str.lower(),
                latency_ms=lat,
                model_version=ver,
                metadata=metadata or {},
            )

            self._records.append(rec)
            if len(self._records) > self.max_records:
                self._records.pop(0)
            return rec

    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        label: Optional[str] = None,
        modality: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[PredictionRecord]:
        """Query prediction history with filtering parameters."""
        with self._lock:
            results = list(self._records)

        if start_time is not None:
            results = [r for r in results if r.timestamp >= start_time]
        if end_time is not None:
            results = [r for r in results if r.timestamp <= end_time]
        if label is not None:
            lbl_upper = label.upper()
            results = [r for r in results if r.label == lbl_upper]
        if modality is not None:
            mod_lower = modality.lower()
            results = [r for r in results if r.modality == mod_lower]
        if min_confidence is not None:
            results = [r for r in results if r.confidence >= min_confidence]

        if limit is not None and limit > 0:
            results = results[-limit:]

        return results

    def export_json(self, filepath: Union[str, Path]) -> Path:
        """Export all current records to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [asdict(r) for r in self._records]
        path.write_text(json.dumps(data, indent=2))
        return path

    def export_csv(self, filepath: Union[str, Path]) -> Path:
        """Export all current records to CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)

        fields = ["id", "timestamp", "confidence", "label", "modality", "latency_ms", "model_version"]
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
        return path

    def clear(self) -> None:
        """Clear all stored prediction history."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        """Get record count."""
        with self._lock:
            return len(self._records)
