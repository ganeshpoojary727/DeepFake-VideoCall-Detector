"""
Microphone capture service with real-time audio streaming and WASAPI system loopback support.

Architecture
------------
* Supports both WASAPI System Audio Loopback capture (remote caller's audio from speakers)
  and default microphone capture.
* `pyaudiowpatch` or `sounddevice` input callback writes chunks into a thread-safe ring buffer.
* A background `threading.Thread` reads from the ring buffer, applies Voice Activity Detection (VAD),
  and segments speech into overlapping windows.
* Published on `EventBus` as `AudioLevelEvent` for GUI waveform display.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from app.config.settings import settings
from app.services.event_bus import AudioLevelEvent, ServiceStateEvent, event_bus
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Ring Buffer
# ──────────────────────────────────────────────


class RingBuffer:
    """Fixed-capacity thread-safe circular audio buffer."""

    def __init__(self, capacity_seconds: float = 5.0, sample_rate: int = 16000) -> None:
        self._capacity = int(capacity_seconds * sample_rate)
        self._buffer: np.ndarray = np.zeros(self._capacity, dtype=np.float32)
        self._write_pos: int = 0
        self._samples_written: int = 0
        self._lock = threading.Lock()

    def write(self, audio: np.ndarray) -> None:
        """Append audio samples, wrapping around on overflow."""
        n = len(audio)
        if n == 0:
            return
        with self._lock:
            end = (self._write_pos + n) % self._capacity
            if end > self._write_pos:
                self._buffer[self._write_pos:end] = audio
            else:
                first = self._capacity - self._write_pos
                self._buffer[self._write_pos:] = audio[:first]
                self._buffer[:end] = audio[first:]
            self._write_pos = end
            self._samples_written += n

    def read_latest(self, n_samples: int) -> Optional[np.ndarray]:
        """Return the most recent *n_samples* as a copy, or ``None`` if insufficient."""
        with self._lock:
            if self._samples_written < n_samples:
                return None
            start = (self._write_pos - n_samples) % self._capacity
            if start + n_samples <= self._capacity:
                return self._buffer[start:start + n_samples].copy()
            else:
                first = self._capacity - start
                return np.concatenate([
                    self._buffer[start:].copy(),
                    self._buffer[:n_samples - first].copy(),
                ])


# ──────────────────────────────────────────────
# Energy-Based VAD
# ──────────────────────────────────────────────


def _is_speech(audio: np.ndarray, threshold: float = 0.003) -> bool:
    """Return True if the RMS energy of the audio exceeds the threshold."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms > threshold


# ──────────────────────────────────────────────
# Microphone / WASAPI Loopback Service
# ──────────────────────────────────────────────


class MicrophoneService:
    """
    Real-time audio capture service supporting both System Audio Loopback (WASAPI)
    and standard microphone capture.

    Parameters
    ----------
    sample_rate : int
        Target sample rate (Hz).
    window_seconds : float
        Length of each audio segment sent to the callback.
    hop_seconds : float
        Sliding window hop size.
    audio_source : str
        "loopback" (WASAPI system audio, what you HEAR from caller) or "mic".
    on_audio : callable | None
        Called with (audio: np.ndarray, sr: int) for each speech window.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        audio_source: Optional[str] = None,
        on_audio: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(window_seconds * sample_rate)
        self.hop_samples = int(hop_seconds * sample_rate)
        self.audio_source = audio_source or getattr(settings.audio, "audio_source", "loopback")
        self.on_audio = on_audio

        self._ring = RingBuffer(capacity_seconds=10.0, sample_rate=sample_rate)
        self._stream = None
        self._pyaudio_obj = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start audio capture stream (WASAPI loopback or mic fallback)."""
        if self._running:
            logger.warning("MicrophoneService already running")
            return

        self._stop_event.clear()
        success = False

        # Mode A: Try WASAPI Loopback Capture via pyaudiowpatch if requested
        if self.audio_source == "loopback" and sys.platform == "win32":
            success = self._start_wasapi_loopback()

        # Mode B: Fallback to standard sounddevice mic capture if loopback fails or source == "mic"
        if not success:
            logger.info("Using standard Microphone capture (audio_source=%s)", self.audio_source)
            success = self._start_standard_mic()

        if not success:
            logger.error("Failed to initialize any audio capture stream.")
            return

        self._worker = threading.Thread(
            target=self._processing_loop,
            name="MicrophoneWorker",
            daemon=True,
        )
        self._worker.start()
        self._running = True

        event_bus.publish(ServiceStateEvent(service="MicrophoneService", running=True))
        logger.info(
            "MicrophoneService started (source=%s, sr=%dHz, window=%.1fs)",
            self.audio_source,
            self.sample_rate,
            self.window_samples / self.sample_rate,
        )

    def _start_wasapi_loopback(self) -> bool:
        """Attempt WASAPI System Loopback capture via pyaudiowpatch."""
        try:
            import pyaudiowpatch as pyaudio

            self._pyaudio_obj = pyaudio.PyAudio()
            wasapi_info = self._pyaudio_obj.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self._pyaudio_obj.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )

            loopback_device = None
            for dev in self._pyaudio_obj.get_loopback_device_info_generator():
                if default_speakers["name"] in dev["name"]:
                    loopback_device = dev
                    break

            if loopback_device is None:
                loopbacks = list(self._pyaudio_obj.get_loopback_device_info_generator())
                if loopbacks:
                    loopback_device = loopbacks[0]

            if loopback_device is None:
                logger.warning("No WASAPI loopback device found.")
                return False

            dev_sr = int(loopback_device["defaultSampleRate"])
            channels = loopback_device["maxInputChannels"]

            def _wasapi_cb(in_data, frame_count, time_info, status):
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels).mean(axis=1)

                # Resample to self.sample_rate if needed
                if dev_sr != self.sample_rate and len(audio_data) > 0:
                    x_old = np.linspace(0, 1, len(audio_data))
                    x_new = np.linspace(0, 1, int(len(audio_data) * self.sample_rate / dev_sr))
                    audio_data = np.interp(x_new, x_old, audio_data).astype(np.float32)

                self._ring.write(audio_data)
                return (None, pyaudio.paContinue)

            self._stream = self._pyaudio_obj.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=dev_sr,
                input=True,
                input_device_index=loopback_device["index"],
                stream_callback=_wasapi_cb,
            )
            self._stream.start_stream()
            logger.info("Successfully started WASAPI Loopback stream on '%s'", loopback_device["name"])
            return True

        except Exception as exc:
            logger.warning("WASAPI Loopback initialization failed: %s; falling back to mic", exc)
            if self._pyaudio_obj:
                try:
                    self._pyaudio_obj.terminate()
                except Exception:
                    pass
                self._pyaudio_obj = None
            return False

    def _start_standard_mic(self) -> bool:
        """Start standard microphone stream using sounddevice."""
        try:
            import sounddevice as sd

            def _sd_cb(indata: np.ndarray, frames: int, time_info, status) -> None:
                mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
                self._ring.write(mono.copy())

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(0.05 * self.sample_rate),
                callback=_sd_cb,
            )
            self._stream.start()
            return True
        except Exception as exc:
            logger.error("Failed to open sounddevice input stream: %s", exc)
            return False

    def stop(self) -> None:
        """Stop audio stream and background processing thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._stream is not None:
            try:
                if hasattr(self._stream, "stop_stream"):
                    self._stream.stop_stream()
                    self._stream.close()
                else:
                    self._stream.stop()
                    self._stream.close()
            except Exception as exc:
                logger.debug("Error closing stream: %s", exc)
            self._stream = None

        if self._pyaudio_obj is not None:
            try:
                self._pyaudio_obj.terminate()
            except Exception:
                pass
            self._pyaudio_obj = None

        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None

        self._running = False
        event_bus.publish(ServiceStateEvent(service="MicrophoneService", running=False))
        logger.info("MicrophoneService stopped")

    def _processing_loop(self) -> None:
        """Read ring buffer, apply VAD, deliver speech windows to callback."""
        last_rms_publish = time.monotonic()
        hop_interval = self.hop_samples / self.sample_rate

        while not self._stop_event.wait(timeout=hop_interval):
            now = time.monotonic()
            if now - last_rms_publish >= 0.1:
                chunk = self._ring.read_latest(int(0.1 * self.sample_rate))
                if chunk is not None:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    event_bus.publish(AudioLevelEvent(level=min(rms * 10, 1.0)))
                last_rms_publish = now

            window = self._ring.read_latest(self.window_samples)
            if window is None:
                continue

            if not _is_speech(window):
                continue

            if self.on_audio is not None:
                try:
                    self.on_audio(window, self.sample_rate)
                except Exception as exc:
                    logger.error("Error in on_audio callback: %s", exc)
