"""
Voice Detector — in-memory buffer-based audio deepfake detection.

Operates entirely on in-memory NumPy arrays (no disk I/O).
Converts raw audio waveform → 128-bin log-Mel spectrogram → PyTorch tensor
→ model inference → fake probability (0.0 to 1.0).

Falls back to spectral heuristic analysis when no trained model weights
are available, so the system remains functional during development.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio.transforms as T

from app.audio.models.cnn_model import AudioDeepfakeCNN, DeepFakeCNN
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceDetector:
    """
    Audio deepfake detector operating on in-memory buffers.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to trained model checkpoint. Defaults to ``settings.MODEL_SAVE_PATH``.
    device : torch.device, optional
        Compute device. Defaults to ``settings.DEVICE``.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or settings.DEVICE
        self._model: Optional[torch.nn.Module] = None
        self._model_loaded = False
        self._use_sigmoid_model = False

        # Mel-spectrogram transform (128 bins)
        self._mel_transform = T.MelSpectrogram(
            sample_rate=settings.audio.sample_rate,
            n_fft=settings.audio.n_fft,
            hop_length=settings.audio.hop_length,
            n_mels=settings.audio.n_mels,
        )

        # Load model if available
        model_path = Path(model_path or settings.MODEL_SAVE_PATH)
        if model_path.exists():
            try:
                self._load_model(model_path)
            except Exception as exc:
                logger.warning("VoiceDetector: model load failed: %s", exc)
        else:
            logger.info(
                "VoiceDetector: no checkpoint at %s — using heuristic analysis",
                model_path,
            )

    def _load_model(self, path: Path) -> None:
        """Load model weights, trying AudioDeepfakeCNN first, then DeepFakeCNN."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)

        # Try AudioDeepfakeCNN (sigmoid output) first
        try:
            model = AudioDeepfakeCNN()
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self._model = model
            self._model_loaded = True
            self._use_sigmoid_model = True
            logger.info("VoiceDetector: loaded AudioDeepfakeCNN from %s", path)
            return
        except Exception:
            pass

        # Fall back to DeepFakeCNN (2-class logits)
        try:
            model = DeepFakeCNN(num_classes=settings.model.num_classes)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self._model = model
            self._model_loaded = True
            self._use_sigmoid_model = False
            logger.info("VoiceDetector: loaded DeepFakeCNN from %s", path)
        except Exception as exc:
            logger.warning("VoiceDetector: failed to load any model: %s", exc)

    def predict_from_buffer(
        self, audio_np: np.ndarray, sr: int = 16000
    ) -> float:
        """
        Predict deepfake probability from an in-memory audio buffer.

        Parameters
        ----------
        audio_np : np.ndarray
            Raw audio waveform, shape ``(samples,)``, dtype float32.
        sr : int
            Sample rate of the audio (default: 16000).

        Returns
        -------
        float
            Fake probability from 0.0 (definitely real) to 1.0 (definitely fake).
        """
        if len(audio_np) < sr:  # Less than 1 second
            logger.debug("Audio buffer too short (%d samples), returning 0.5", len(audio_np))
            return 0.5

        start = time.perf_counter()

        if self._model_loaded and self._model is not None:
            score = self._predict_with_model(audio_np, sr)
        else:
            score = self._predict_heuristic(audio_np, sr)

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug("VoiceDetector inference: score=%.4f, latency=%.1fms", score, elapsed)
        return score

    def _predict_with_model(self, audio_np: np.ndarray, sr: int) -> float:
        """Run model inference on audio buffer."""
        # Convert to tensor
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, samples)

        # Compute log-Mel spectrogram
        mel_spec = self._mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-9)  # (1, n_mels, time_frames)

        # Pad or truncate time dimension to target_length
        target_len = settings.audio.target_length
        if log_mel.shape[-1] > target_len:
            log_mel = log_mel[:, :, :target_len]
        elif log_mel.shape[-1] < target_len:
            pad_size = target_len - log_mel.shape[-1]
            log_mel = torch.nn.functional.pad(log_mel, (0, pad_size))

        # Add channel dimension: (1, 1, n_mels, time_frames)
        tensor = log_mel.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(tensor)

            if self._use_sigmoid_model:
                # AudioDeepfakeCNN: output is (batch, 1) sigmoid probability
                score = output.squeeze().item()
            else:
                # DeepFakeCNN: output is (batch, num_classes) logits
                probs = torch.softmax(output, dim=1)
                score = probs[0, 1].item()  # P(fake)

        return float(np.clip(score, 0.0, 1.0))

    def _predict_heuristic(self, audio_np: np.ndarray, sr: int) -> float:
        """
        Heuristic-based analysis when no trained model is available.

        Analyzes spectral characteristics that differ between real and
        deepfake audio:
        - Spectral flatness (real speech has more tonal structure)
        - High-frequency energy ratio (deepfakes often lack HF content)
        - Zero-crossing rate variance (deepfakes tend to be smoother)
        """
        try:
            # Spectral flatness via FFT
            fft = np.fft.rfft(audio_np)
            magnitude = np.abs(fft)
            magnitude = magnitude[magnitude > 0]

            if len(magnitude) == 0:
                return 0.5

            # Geometric mean / arithmetic mean (Wiener entropy)
            log_mean = np.mean(np.log(magnitude + 1e-10))
            geo_mean = np.exp(log_mean)
            arith_mean = np.mean(magnitude)
            spectral_flatness = geo_mean / (arith_mean + 1e-10)

            # High-frequency energy ratio
            n_bins = len(magnitude)
            hf_cutoff = int(n_bins * 0.7)
            hf_energy = np.sum(magnitude[hf_cutoff:] ** 2)
            total_energy = np.sum(magnitude ** 2) + 1e-10
            hf_ratio = hf_energy / total_energy

            # Zero-crossing rate variance
            zcr = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2 * len(audio_np))

            # Combine heuristics
            # High spectral flatness + low HF + low ZCR variance → more likely fake
            fake_indicators = 0.0
            if spectral_flatness > 0.4:
                fake_indicators += 0.3
            if hf_ratio < 0.05:
                fake_indicators += 0.3
            if zcr < 0.02:
                fake_indicators += 0.2

            score = np.clip(fake_indicators + 0.1, 0.0, 1.0)
            return float(score)

        except Exception as exc:
            logger.debug("Heuristic analysis failed: %s", exc)
            return 0.5

    @property
    def is_ready(self) -> bool:
        """Whether the detector is operational (model or heuristic)."""
        return True  # Always ready — falls back to heuristics