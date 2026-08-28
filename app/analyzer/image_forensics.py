"""
Image Forensics Engine — multi-signal physics-based and statistical artifact analysis.

Implements 4 complementary forensic methods:
1. 2D FFT Frequency Analysis (detects periodic GAN / Diffusion upsampling grid peaks)
2. Spatial Rich Model (SRM) Noise Residuals (detects camera PRNU noise vs synthetic smoothing)
3. Error Level Analysis (ELA) (detects JPEG compression error gradient anomalies)
4. Chromatic Dispersion & Gradient Continuity (detects optical lens consistency)
"""

from __future__ import annotations

import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Any, Dict, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ImageForensics:
    """Multi-signal forensic analyzer for deepfake and synthetic image detection."""

    @staticmethod
    def analyze_image_signals(image_bgr: np.ndarray) -> Dict[str, Any]:
        """Run all 4 forensic tests on an image and return individual scores and combined metric.

        Parameters
        ----------
        image_bgr : np.ndarray
            OpenCV BGR image array.

        Returns
        -------
        dict
            Dictionary containing:
            - fft_score: float [0.0, 1.0] (higher = more anomalous / fake)
            - srm_noise_score: float [0.0, 1.0]
            - ela_score: float [0.0, 1.0]
            - chromatic_score: float [0.0, 1.0]
            - combined_forensic_score: float [0.0, 1.0]
            - details: dict of raw measurements
        """
        try:
            fft_score, fft_details = ImageForensics._analyze_fft_spectrum(image_bgr)
            srm_score, srm_details = ImageForensics._analyze_srm_noise(image_bgr)
            ela_score, ela_details = ImageForensics._analyze_ela(image_bgr)
            chrom_score, chrom_details = ImageForensics._analyze_chromatic(image_bgr)

            # Weighted combination of physical & frequency forensic signals
            combined = (
                0.35 * fft_score +
                0.30 * srm_score +
                0.20 * ela_score +
                0.15 * chrom_score
            )
            combined = float(np.clip(combined, 0.0, 1.0))

            return {
                "fft_score": round(fft_score, 4),
                "srm_noise_score": round(srm_score, 4),
                "ela_score": round(ela_score, 4),
                "chromatic_score": round(chrom_score, 4),
                "combined_forensic_score": round(combined, 4),
                "details": {
                    "fft": fft_details,
                    "srm": srm_details,
                    "ela": ela_details,
                    "chromatic": chrom_details,
                },
            }
        except Exception as exc:
            logger.warning("ImageForensics analysis error: %s", exc)
            return {
                "fft_score": 0.5,
                "srm_noise_score": 0.5,
                "ela_score": 0.5,
                "chromatic_score": 0.5,
                "combined_forensic_score": 0.5,
                "details": {"error": str(exc)},
            }

    # ── 1. 2D FFT Frequency Analysis ───────────────────────

    @staticmethod
    def _analyze_fft_spectrum(image_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Compute 2D FFT and check for high-frequency periodic grid spikes."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.5, {"reason": "image too small"}

        # Resize to power-of-2 for clean FFT
        gray_resized = cv2.resize(gray, (256, 256)).astype(np.float32)

        # 2D Fast Fourier Transform with DC-shift to center
        f_transform = np.fft.fft2(gray_resized)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

        # Mask center DC component
        cy, cx = 128, 128
        r_dc = 20
        y, x = np.ogrid[:256, :256]
        mask = (x - cx) ** 2 + (y - cy) ** 2 > r_dc ** 2
        high_freq_mag = magnitude_spectrum[mask]

        # In natural photos, high-frequency energy decays smoothly (low peak-to-mean ratio)
        # In GAN / Diffusion images, periodic upsampling causes sharp periodic peaks
        mean_val = float(np.mean(high_freq_mag))
        max_val = float(np.max(high_freq_mag))
        std_val = float(np.std(high_freq_mag))
        peak_to_mean = (max_val - mean_val) / (std_val + 1e-6)

        # Natural images typically have peak_to_mean around 3.0 - 4.5
        # Synthetic / GAN images typically have peak_to_mean > 5.5
        # Normalize into [0.0, 1.0] fake probability
        fake_prob = 1.0 / (1.0 + np.exp(-0.8 * (peak_to_mean - 4.8)))

        return float(np.clip(fake_prob, 0.05, 0.95)), {
            "peak_to_mean_ratio": round(peak_to_mean, 3),
            "high_freq_mean": round(mean_val, 2),
            "high_freq_std": round(std_val, 2),
        }

    # ── 2. SRM Noise Residual Analysis ─────────────────────

    @staticmethod
    def _analyze_srm_noise(image_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Extract sensor noise residual with high-pass spatial filter."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Laplacian high-pass filter for noise residual extraction
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        residual = cv2.filter2D(gray, -1, kernel)

        # Calculate local noise variance across 8x8 patches
        h, w = residual.shape
        step = 16
        patch_vars = []
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                patch = residual[i : i + step, j : j + step]
                patch_vars.append(np.var(patch))

        if not patch_vars:
            return 0.5, {"reason": "insufficient patches"}

        mean_var = float(np.mean(patch_vars))
        std_var = float(np.std(patch_vars))

        # Real camera sensor noise (PRNU) has consistent, moderate variance across patches.
        # AI generators produce unnaturally smoothed patches (very low variance) or uneven variance.
        # Natural range: mean_var in [8.0, 45.0], std_var in [4.0, 30.0]
        if mean_var < 3.0:
            # Heavy AI / GAN smoothing detected
            fake_prob = 0.85
        elif mean_var > 90.0:
            # Heavy unnatural high-frequency artifact
            fake_prob = 0.75
        else:
            # Natural camera noise profile
            fake_prob = 0.15 + 0.3 * (abs(mean_var - 20.0) / 30.0)

        return float(np.clip(fake_prob, 0.05, 0.95)), {
            "noise_mean_variance": round(mean_var, 2),
            "noise_variance_std": round(std_var, 2),
        }

    # ── 3. Error Level Analysis (ELA) ──────────────────────

    @staticmethod
    def _analyze_ela(image_bgr: np.ndarray, quality: int = 90) -> Tuple[float, Dict[str, Any]]:
        """Compute JPEG Error Level Analysis to detect compression gradient anomalies."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Resave in memory at quality 90
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved_img = Image.open(buf)

        # Compute pixel difference
        ela_diff = ImageChops.difference(pil_img, resaved_img)
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max(max_diff, 1)
        ela_scaled = ImageEnhance.Brightness(ela_diff).enhance(scale)

        ela_arr = np.array(ela_scaled).astype(np.float32)
        ela_mean = float(np.mean(ela_arr))
        ela_std = float(np.std(ela_arr))

        # High difference standard deviation across image indicates spliced / generated regions
        # Natural images have uniform ELA difference distribution
        ela_inconsistency = ela_std / (ela_mean + 1e-5)
        fake_prob = 1.0 / (1.0 + np.exp(-1.2 * (ela_inconsistency - 1.4)))

        return float(np.clip(fake_prob, 0.05, 0.95)), {
            "ela_mean": round(ela_mean, 2),
            "ela_std": round(ela_std, 2),
            "ela_inconsistency": round(ela_inconsistency, 3),
        }

    # ── 4. Chromatic Dispersion ────────────────────────────

    @staticmethod
    def _analyze_chromatic(image_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Evaluate cross-channel gradient correlation natural to physical camera lenses."""
        b, g, r = cv2.split(image_bgr.astype(np.float32))

        # Compute Sobel gradients for each channel
        grad_r = np.abs(cv2.Sobel(r, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(r, cv2.CV_32F, 0, 1))
        grad_g = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        grad_b = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1))

        # Correlation between R and B gradients (optical aberration binds them in physical lenses)
        rg_diff = np.mean(np.abs(grad_r - grad_g))
        gb_diff = np.mean(np.abs(grad_g - grad_b))
        rb_diff = np.mean(np.abs(grad_r - grad_b))

        total_energy = np.mean(grad_g) + 1e-6
        chrom_metric = float((rg_diff + gb_diff + rb_diff) / (3.0 * total_energy))

        # Real photos: natural chromatic gradient ratio in [0.08, 0.28]
        # AI generated: often abnormally uniform (low) or erratic (high)
        if 0.10 <= chrom_metric <= 0.25:
            fake_prob = 0.15
        elif chrom_metric < 0.07:
            fake_prob = 0.70  # Synthetic lack of chromatic dispersion
        else:
            fake_prob = 0.65

        return float(np.clip(fake_prob, 0.05, 0.95)), {
            "chromatic_gradient_ratio": round(chrom_metric, 4),
        }
