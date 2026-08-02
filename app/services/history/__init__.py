"""History service package exports."""

from app.services.history.history_service import HistoryService, PredictionRecord

__all__ = [
    "HistoryService",
    "PredictionRecord",
]
