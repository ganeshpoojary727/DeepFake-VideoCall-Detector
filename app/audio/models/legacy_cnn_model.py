"""Legacy CNN models compatibility re-export module."""

from app.audio.models.legacy.legacy_cnn import AudioDeepfakeCNN, DeepFakeCNN, LightCNN

__all__ = [
    "DeepFakeCNN",
    "LightCNN",
    "AudioDeepfakeCNN",
]
