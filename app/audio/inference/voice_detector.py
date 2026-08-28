"""
Voice Detector — High-precision offline audio deepfake detector using AASIST.

Performs multi-window temporal chunking, spectral forensic cue extraction,
and returns structured forensic telemetry (verdict, confidence, raw scores,
spectral artifact ranges, and chunk timeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch

from app.audio.constants.audio_constants import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_RATE,
    LABEL_BONAFIDE,
    LABEL_SPOOF,
)
from app.audio.models.aasist import AASIST
from app.audio.preprocessing.audio_loader import AudioLoader
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceDetector:
    """AASIST-powered audio deepfake detector with multi-window timeline and spectral diagnostics."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        target_sr: int = DEFAULT_SAMPLE_RATE,
        chunk_samples: int = DEFAULT_NUM_SAMPLES,
    ) -> None:
        self.device = device or settings.DEVICE
        self.target_sr = target_sr
        self.chunk_samples = chunk_samples
        self.loader = AudioLoader(target_sr=target_sr, target_samples=chunk_samples)
        self._model: Optional[torch.nn.Module] = None
        self._model_loaded = False

        resolved_path = Path(model_path or settings.MODEL_SAVE_PATH)
        if resolved_path.exists():
            try:
                self._load_model(resolved_path)
            except Exception as exc:
                logger.warning("VoiceDetector: Checkpoint load failed (%s) — initializing fresh AASIST", exc)
                self._init_default_model()
        else:
            logger.info("VoiceDetector: No checkpoint at %s — initializing default AASIST", resolved_path)
            self._init_default_model()

    def _init_default_model(self) -> None:
        """Initialize an unweighted/default AASIST architecture for testing/fallback."""
        self._model = AASIST(num_classes=settings.model.num_classes).to(self.device)
        self._model.eval()
        self._model_loaded = True

    def _load_model(self, path: Path) -> None:
        """Load AASIST model state dict from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model = AASIST(num_classes=settings.model.num_classes)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        self._model = model
        self._model_loaded = True
        logger.info("VoiceDetector: Loaded AASIST checkpoint from %s", path)

    # ── Inference APIs ─────────────────────────────────────────────────────────

    def predict_buffer(self, audio_buffer: np.ndarray) -> float:
        """Legacy buffer-based inference returning single spoof probability in [0.0, 1.0].

        Parameters
        ----------
        audio_buffer : np.ndarray
            1D audio sample array.

        Returns
        -------
        float
            Deepfake spoof probability.
        """
        if self._model is None:
            return 0.5

        waveform = self.loader.pad_crop_waveform(audio_buffer, target_samples=self.chunk_samples)
        tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self._model(tensor)
            if isinstance(out, tuple):
                logits = out[1] if len(out) > 1 else out[0]
            else:
                logits = out
            probs = torch.softmax(logits, dim=-1)[0]
            spoof_prob = float(probs[1].item() if probs.shape[0] > 1 else probs[0].item())

        return float(np.clip(spoof_prob, 0.0, 1.0))

    def predict_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Perform offline deepfake analysis on an uploaded audio file.

        Parameters
        ----------
        file_path : str | Path
            Path to candidate audio file.

        Returns
        -------
        Dict[str, Any]
            Structured telemetry output dictionary.
        """
        audio, sr = self.loader.load_audio(file_path, target_sr=self.target_sr, normalize=True)
        return self.predict_detailed(audio, sr=sr)

    def predict_detailed(
        self,
        waveform: np.ndarray,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform full chunked timeline analysis and spectral artifact extraction.

        Parameters
        ----------
        waveform : np.ndarray
            1D audio array.
        sr : int | None
            Waveform sample rate (default: 16kHz).

        Returns
        -------
        Dict[str, Any]
            Structured forensic output containing verdict, confidence, raw_scores,
            spectral_cues, and timeline.
        """
        sample_rate = sr or self.target_sr

        # 1. Spectral Cues Forensic Extraction
        spectral_cues = self.extract_spectral_cues(waveform, sr=sample_rate)

        # 2. Multi-Window Temporal Chunking
        chunks = self.loader.chunk_waveform(
            waveform,
            chunk_samples=self.chunk_samples,
            hop_samples=self.chunk_samples // 2,
            sr=sample_rate,
        )

        timeline: List[Dict[str, Any]] = []
        chunk_spoof_probs: List[float] = []

        for idx, (chunk_wav, start_sec, end_sec) in enumerate(chunks):
            tensor = torch.from_numpy(chunk_wav).float().unsqueeze(0).to(self.device)

            if self._model is not None:
                with torch.no_grad():
                    out = self._model(tensor)
                    logits = out[1] if isinstance(out, tuple) and len(out) > 1 else (out[0] if isinstance(out, tuple) else out)
                    probs = torch.softmax(logits, dim=-1)[0]
                    c_spoof = float(probs[1].item() if probs.shape[0] > 1 else probs[0].item())
                    c_bonafide = float(probs[0].item() if probs.shape[0] > 1 else 1.0 - c_spoof)
            else:
                c_spoof = 0.5
                c_bonafide = 0.5

            c_spoof = float(np.clip(c_spoof, 0.0001, 0.9999))
            c_bonafide = float(round(1.0 - c_spoof, 4))
            chunk_spoof_probs.append(c_spoof)

            c_verdict = "FAKE" if c_spoof >= 0.5 else "REAL"
            timeline.append({
                "chunk_index": idx,
                "start_time_sec": round(start_sec, 2),
                "end_time_sec": round(end_sec, 2),
                "spoof_prob": round(c_spoof, 4),
                "bonafide_prob": round(c_bonafide, 4),
                "verdict": c_verdict,
            })

        # 3. Aggregation & Decision Logic
        if chunk_spoof_probs:
            # Weighted mix of mean and max spoof probability for robust anomaly sensitivity
            mean_prob = float(np.mean(chunk_spoof_probs))
            max_prob = float(np.max(chunk_spoof_probs))
            overall_spoof = float(0.6 * mean_prob + 0.4 * max_prob)
        else:
            overall_spoof = 0.5

        overall_spoof = float(np.clip(overall_spoof, 0.0001, 0.9999))
        overall_bonafide = float(round(1.0 - overall_spoof, 4))

        if overall_spoof >= 0.5:
            verdict = "FAKE"
            confidence = overall_spoof
        else:
            verdict = "REAL"
            confidence = overall_bonafide

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "raw_scores": {
                "bonafide_prob": round(float(overall_bonafide), 4),
                "spoof_prob": round(float(overall_spoof), 4),
            },
            "spectral_cues": spectral_cues,
            "timeline": timeline,
        }

    # ── Spectral Forensics ─────────────────────────────────────────────────────

    @staticmethod
    def extract_spectral_cues(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """Extract spectral forensic cues indicative of synthesis and vocoder artifacts.

        Checks for:
        - High-frequency cutoff / truncation (common in Tacotron/FastSpeech + MelGAN/HiFi-GAN pipelines).
        - Robotic synthesis bands / harmonic spikes in upper speech bands (2.5kHz - 6.0kHz).
        - High-frequency energy ratio and spectral flatness.

        Parameters
        ----------
        waveform : np.ndarray
            1D audio array.
        sr : int
            Sampling rate in Hz (default: 16kHz).

        Returns
        -------
        Dict[str, Any]
            Spectral cues and peak artifact frequency ranges.
        """
        if len(waveform) < 512:
            return {
                "peak_artifact_ranges": [],
                "spectral_rolloff_hz": float(sr / 2),
                "high_freq_energy_ratio": 0.5,
                "spectral_flatness": 0.0,
                "artifacts_detected": [],
            }

        # 1. Compute Power Spectrum
        n_fft = min(2048, len(waveform))
        hop_length = n_fft // 4
        stft = np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))
        power_spec = np.mean(stft ** 2, axis=1)  # Average power per frequency bin
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        total_power = float(np.sum(power_spec)) + 1e-12

        # 2. Spectral Roll-off (85% and 95% energy)
        cum_power = np.cumsum(power_spec)
        rolloff_95_idx = np.searchsorted(cum_power, 0.95 * total_power)
        rolloff_95_hz = float(freqs[min(rolloff_95_idx, len(freqs) - 1)])

        # 3. High-Frequency Energy Ratio (energy > 6kHz / total energy)
        hf_mask = freqs >= 6000.0
        hf_power = float(np.sum(power_spec[hf_mask]))
        hf_ratio = float(hf_power / total_power)

        # 4. Mid-Band Robotic Synthesis Band Analysis (2.5kHz - 6kHz)
        mid_mask = (freqs >= 2500.0) & (freqs <= 6000.0)
        mid_power = power_spec[mid_mask]
        if len(mid_power) > 0 and np.mean(mid_power) > 0:
            # Spectral flatness in mid-band: geom_mean / arith_mean
            log_power = np.log(mid_power + 1e-12)
            geom_mean = np.exp(np.mean(log_power))
            arith_mean = np.mean(mid_power)
            spectral_flatness = float(geom_mean / (arith_mean + 1e-12))
        else:
            spectral_flatness = 0.0

        # 5. Identify Artifact Frequency Ranges
        peak_artifact_ranges: List[Dict[str, Any]] = []
        artifacts_detected: List[str] = []

        # High-Frequency Truncation check (e.g. sharp cutoff below 7.2kHz in 16kHz audio)
        if rolloff_95_hz < 7200.0 or hf_ratio < 0.008:
            cutoff_freq = round(rolloff_95_hz, 0)
            peak_artifact_ranges.append({
                "range_hz": [int(cutoff_freq), int(sr // 2)],
                "type": "high_frequency_truncation",
                "severity": "high" if hf_ratio < 0.002 else "medium",
                "description": f"Steep spectral attenuation above {int(cutoff_freq)}Hz typical of neural vocoders/TTS",
            })
            artifacts_detected.append("high_frequency_truncation")

        # Robotic Synthesis Band check (abnormal harmonic peaks / flatness in 2.5kHz - 5.5kHz)
        if spectral_flatness > 0.45 or (hf_ratio < 0.015 and spectral_flatness > 0.35):
            peak_artifact_ranges.append({
                "range_hz": [2500, 5500],
                "type": "robotic_synthesis_bands",
                "severity": "medium",
                "description": "Unnatural harmonic resonance and spectral flatness in synthesis band",
            })
            artifacts_detected.append("robotic_synthesis_bands")

        return {
            "peak_artifact_ranges": peak_artifact_ranges,
            "spectral_rolloff_hz": round(rolloff_95_hz, 1),
            "high_freq_energy_ratio": round(hf_ratio, 4),
            "spectral_flatness": round(spectral_flatness, 4),
            "artifacts_detected": artifacts_detected,
        }