"""
Audio Loader & Protocol Parser module for ASVspoof and AASIST Deepfake Forensics.

Provides high-performance audio loading, resampling to fixed 16kHz, waveform
peak normalization, fixed-length chunking/padding (64,600 samples ~4s for AASIST),
and robust protocol parsing for ASVspoof 2019 (LA/PA) and ASVspoof 2021 (LA/DF).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from app.audio.constants.audio_constants import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_RATE,
    LABEL_BONAFIDE,
    LABEL_BONAFIDE_STR,
    LABEL_SPOOF,
    LABEL_SPOOF_STR,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AudioLoader:
    """Standardized audio loading, resampling, windowing, and protocol parser.

    Parameters
    ----------
    target_sr : int
        Target sample rate in Hz (default: 16,000 Hz).
    target_samples : int
        Target sample length for fixed-window models like AASIST (default: 64,600).
    pad_mode : str
        Padding method when waveform length < target_samples ('wrap', 'zero', or 'reflect').
    """

    def __init__(
        self,
        target_sr: int = DEFAULT_SAMPLE_RATE,
        target_samples: int = DEFAULT_NUM_SAMPLES,
        pad_mode: str = "wrap",
    ) -> None:
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.pad_mode = pad_mode

    # ── Audio Loading & Normalization ──────────────────────────────────────────

    def load_audio(
        self,
        audio_path: Union[str, Path],
        target_sr: Optional[int] = None,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Load an audio file into a 1D mono float32 numpy array.

        Uses soundfile as primary fast decoder, with librosa fallback for broader format support.

        Parameters
        ----------
        audio_path : str | Path
            Path to audio file (.flac, .wav, .mp3, .ogg, .m4a, etc.).
        target_sr : int | None
            Sample rate to resample to. Defaults to self.target_sr (16kHz).
        normalize : bool
            Whether to peak-normalize the waveform to [-1.0, 1.0].

        Returns
        -------
        Tuple[np.ndarray, int]
            (waveform_1d, sample_rate)
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        desired_sr = target_sr or self.target_sr
        audio: Optional[np.ndarray] = None
        sr: Optional[int] = None

        # 1. Primary: Soundfile
        try:
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # Stereo to mono
            if desired_sr is not None and sr != desired_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=desired_sr)
                sr = desired_sr
        except Exception as sf_err:
            logger.debug("Soundfile read failed for %s: %s. Falling back to librosa.", path.name, sf_err)
            try:
                audio, sr = librosa.load(str(path), sr=desired_sr, mono=True)
                audio = audio.astype(np.float32)
            except Exception as lib_err:
                raise RuntimeError(
                    f"Failed to load audio file '{path}' using soundfile and librosa: {lib_err}"
                ) from lib_err

        if audio is None or len(audio) == 0:
            raise ValueError(f"Audio file '{path}' contains no readable audio samples.")

        # Ensure 1D float32
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # 2. Peak Normalization
        if normalize:
            audio = self.normalize_waveform(audio)

        return audio, int(sr or desired_sr)

    @staticmethod
    def normalize_waveform(audio: np.ndarray) -> np.ndarray:
        """Peak-normalize waveform into [-1.0, 1.0]."""
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0:
            return (audio / peak).astype(np.float32)
        return audio.astype(np.float32)

    # ── Waveform Windowing & Padding ──────────────────────────────────────────

    def pad_crop_waveform(
        self,
        waveform: np.ndarray,
        target_samples: Optional[int] = None,
        mode: Optional[str] = None,
        random_crop: bool = False,
    ) -> np.ndarray:
        """Pad or crop a waveform to exact target sample length (e.g. 64,600 samples).

        Parameters
        ----------
        waveform : np.ndarray
            1D audio array.
        target_samples : int | None
            Required length. Defaults to self.target_samples.
        mode : str | None
            'wrap' (repeat waveform), 'zero' (zero pad), or 'reflect'. Defaults to self.pad_mode.
        random_crop : bool
            If True and length > target_samples, randomly crop a window (useful during training).
            If False, slice deterministic front window [0:target_samples].

        Returns
        -------
        np.ndarray
            Array of exact shape (target_samples,) and dtype float32.
        """
        req_len = target_samples or self.target_samples
        pad_mode = mode or self.pad_mode
        curr_len = len(waveform)

        if curr_len == req_len:
            return waveform.astype(np.float32)

        if curr_len < req_len:
            pad_needed = req_len - curr_len
            if pad_mode == "wrap":
                repeat_times = int(math.ceil(req_len / max(curr_len, 1)))
                repeated = np.tile(waveform, repeat_times)
                return repeated[:req_len].astype(np.float32)
            elif pad_mode == "zero":
                return np.pad(waveform, (0, pad_needed), mode="constant", constant_values=0.0).astype(np.float32)
            elif pad_mode == "reflect":
                if curr_len <= 1:
                    return np.pad(waveform, (0, pad_needed), mode="constant", constant_values=0.0).astype(np.float32)
                return np.pad(waveform, (0, pad_needed), mode="reflect").astype(np.float32)
            else:
                # Default wrap
                repeat_times = int(math.ceil(req_len / max(curr_len, 1)))
                repeated = np.tile(waveform, repeat_times)
                return repeated[:req_len].astype(np.float32)

        # curr_len > req_len: crop
        if random_crop:
            start = np.random.randint(0, curr_len - req_len + 1)
        else:
            start = 0

        return waveform[start : start + req_len].astype(np.float32)

    def chunk_waveform(
        self,
        waveform: np.ndarray,
        chunk_samples: Optional[int] = None,
        hop_samples: Optional[int] = None,
        sr: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, float, float]]:
        """Slice an arbitrary-length waveform into overlapping chunks for timeline analysis.

        Parameters
        ----------
        waveform : np.ndarray
            Full audio waveform.
        chunk_samples : int | None
            Chunk window size in samples (default 64,600).
        hop_samples : int | None
            Hop step in samples (default chunk_samples // 2).
        sr : int | None
            Sample rate for calculating time in seconds (default self.target_sr).

        Returns
        -------
        List[Tuple[np.ndarray, float, float]]
            List of (chunk_waveform_64600, start_time_sec, end_time_sec).
        """
        c_samples = chunk_samples or self.target_samples
        h_samples = hop_samples or (c_samples // 2)
        sample_rate = sr or self.target_sr
        curr_len = len(waveform)

        if curr_len <= c_samples:
            chunk = self.pad_crop_waveform(waveform, target_samples=c_samples, mode="wrap")
            duration = float(curr_len / sample_rate)
            return [(chunk, 0.0, duration)]

        chunks: List[Tuple[np.ndarray, float, float]] = []
        start = 0
        while start < curr_len:
            end = start + c_samples
            if end <= curr_len:
                chunk = waveform[start:end]
            else:
                chunk = self.pad_crop_waveform(waveform[start:], target_samples=c_samples, mode="wrap")

            start_sec = float(start / sample_rate)
            end_sec = float(min(end, curr_len) / sample_rate)
            chunks.append((chunk.astype(np.float32), start_sec, end_sec))

            if end >= curr_len:
                break
            start += h_samples

        return chunks

    # ── ASVspoof Protocol Parsing ──────────────────────────────────────────────

    @staticmethod
    def parse_asvspoof_protocol(
        protocol_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """Parse standard ASVspoof 2019 (LA/PA) and ASVspoof 2021 (LA/DF) protocol text files.

        Supported line formats:
        - ASVspoof 2019: `SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT ATTACK_TYPE KEY`
          (e.g., `LA_0079 LA_D_1047731 - - bonafide` or `LA_0079 LA_D_1105538 - A07 spoof`)
        - ASVspoof 2021 LA: `SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT ATTACK_TYPE KEY ...`
        - ASVspoof 2021 DF: `AUDIO_FILE_NAME CODEC SRC ATTACK KEY ...` or `SPEAKER_ID FILE ... KEY`
        - Generic: `<filename> <bonafide|spoof>` or `<filename> <0|1>`

        Parameters
        ----------
        protocol_path : str | Path
            Path to the protocol .txt file.

        Returns
        -------
        List[Dict[str, Any]]
            List of parsed sample metadata dictionaries:
            {
                "file_name": str,
                "label": int (0 for bonafide, 1 for spoof),
                "label_str": "bonafide" | "spoof",
                "speaker_id": str | None,
                "attack_type": str | None,
                "environment": str | None,
            }
        """
        path = Path(protocol_path)
        if not path.exists():
            raise FileNotFoundError(f"Protocol file not found: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_idx, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                # Identify key / label ('bonafide' vs 'spoof' or 0 vs 1)
                label: Optional[int] = None
                label_str: Optional[str] = None

                # Check tokens in reverse for label
                for token in reversed(parts):
                    clean_tok = token.lower().strip()
                    if clean_tok in ("bonafide", "bona-fide", "real", "0"):
                        label = LABEL_BONAFIDE
                        label_str = LABEL_BONAFIDE_STR
                        break
                    elif clean_tok in ("spoof", "fake", "deepfake", "1"):
                        label = LABEL_SPOOF
                        label_str = LABEL_SPOOF_STR
                        break

                if label is None:
                    # Fallback heuristic: check last token
                    last_tok = parts[-1].lower()
                    if "bonafide" in last_tok:
                        label, label_str = LABEL_BONAFIDE, LABEL_BONAFIDE_STR
                    elif "spoof" in last_tok:
                        label, label_str = LABEL_SPOOF, LABEL_SPOOF_STR
                    else:
                        continue

                # Identify filename token
                # In ASVspoof 2019: parts[1] is file_id (e.g. LA_D_1047731), parts[0] is speaker_id (e.g. LA_0079)
                # In ASVspoof 2021 DF: parts[0] is file_id (e.g. DF_E_2000011) or parts[1]
                file_name = parts[1] if len(parts) >= 3 and not parts[0].endswith((".flac", ".wav", ".mp3")) and (parts[1].startswith(("LA_", "DF_", "PA_", "eval_", "train_")) or "." in parts[1]) else parts[0]

                # Strip extension if present for clean ID
                clean_name = Path(file_name).stem

                speaker_id = parts[0] if len(parts) >= 4 and parts[0] != file_name else None
                environment = parts[2] if len(parts) >= 5 else None
                attack_type = parts[3] if len(parts) >= 5 else None

                records.append({
                    "file_name": clean_name,
                    "label": label,
                    "label_str": label_str,
                    "speaker_id": speaker_id,
                    "attack_type": attack_type,
                    "environment": environment,
                })

        logger.info("Parsed %d protocol records from %s", len(records), path.name)
        return records


# Convenience module-level instances & functions
default_audio_loader = AudioLoader()
load_audio = default_audio_loader.load_audio
pad_crop_waveform = default_audio_loader.pad_crop_waveform
chunk_waveform = default_audio_loader.chunk_waveform
parse_asvspoof_protocol = AudioLoader.parse_asvspoof_protocol
