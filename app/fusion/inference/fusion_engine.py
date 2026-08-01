"""
Multimodal Fusion Engine — combines audio and video deepfake scores.

Weighted late fusion:
    combined = (AUDIO_WEIGHT × audio_score) + (VIDEO_WEIGHT × video_score)
    default:  0.60 × audio  +  0.40 × video

Classification:
    REAL     if combined < 0.50
    DEEPFAKE if combined >= 0.50
"""

from __future__ import annotations

from typing import Dict

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalFusion:
    """
    Weighted multimodal fusion for audio + video deepfake detection.

    Parameters
    ----------
    audio_weight : float
        Weight for the audio score (default: 0.60).
    video_weight : float
        Weight for the video score (default: 0.40).
    threshold : float
        Decision boundary (default: 0.50).
    """

    def __init__(
        self,
        audio_weight: float | None = None,
        video_weight: float | None = None,
        threshold: float = 0.50,
    ) -> None:
        self.audio_weight = audio_weight or settings.AUDIO_WEIGHT
        self.video_weight = video_weight or settings.VIDEO_WEIGHT
        self.threshold = threshold

    def evaluate(
        self, audio_score: float, video_score: float
    ) -> Dict[str, object]:
        """
        Compute weighted fusion of audio and video scores.

        Parameters
        ----------
        audio_score : float
            Audio fake probability [0.0, 1.0].
        video_score : float
            Video fake probability [0.0, 1.0].

        Returns
        -------
        dict
            ``{"combined_score": float, "prediction": str,
              "audio_score": float, "video_score": float,
              "audio_weight": float, "video_weight": float}``
        """
        combined = (self.audio_weight * audio_score) + (self.video_weight * video_score)
        combined = max(0.0, min(1.0, combined))  # Clamp to [0, 1]

        prediction = "DEEPFAKE" if combined >= self.threshold else "REAL"

        result = {
            "combined_score": round(combined, 4),
            "prediction": prediction,
            "audio_score": round(audio_score, 4),
            "video_score": round(video_score, 4),
            "audio_weight": self.audio_weight,
            "video_weight": self.video_weight,
        }

        logger.debug(
            "Fusion: audio=%.4f (w=%.2f) + video=%.4f (w=%.2f) → combined=%.4f → %s",
            audio_score, self.audio_weight,
            video_score, self.video_weight,
            combined, prediction,
        )

        return result
