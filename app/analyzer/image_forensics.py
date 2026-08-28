"""
Visual and Classical Image Forensics Engine.

Implements physics-based, statistical, and frequency domain forensic analysis:
1. Error Level Analysis (ELA): JPEG compression artifact gradient discrepancy.
2. 2D FFT Frequency Analysis: Azimuthal radial power decay & periodic upsampling grid anomalies.
3. Boundary Blending & Noise Residuals: Laplacian variance & edge gradient step across facial margins.
4. Chromatic Dispersion & Spatial Rich Model (SRM) PRNU sensor noise consistency.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ImageForensics:
    """Multi-signal forensic extractor for digital image and video frame manipulation."""

    @classmethod
    def extract_visual_cues(
        cls,
        image_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """Extract standardized classical visual forensic cues for forensic telemetry.

        Parameters
        ----------
        image_bgr : np.ndarray
            OpenCV BGR image frame.
        bbox : Tuple[int, int, int, int] | None
            Optional facial bounding box (x1, y1, x2, y2).

        Returns
        -------
        Dict[str, Any]
            Standardized dictionary containing:
            - ela_discrepancy_score: float [0.0, 1.0]
            - fft_spectral_anomaly: float [0.0, 1.0]
            - boundary_inconsistency: float [0.0, 1.0]
            - combined_score: float [0.0, 1.0]
            - details: dict of detailed measurements
        """
        try:
            ela_score, ela_map, ela_details = cls.compute_ela(image_bgr)
            fft_score, fft_details = cls.compute_fft_spectrum(image_bgr)
            boundary_score, boundary_details = cls.compute_boundary_inconsistency(image_bgr, bbox=bbox)
            srm_score, srm_details = cls.compute_srm_noise(image_bgr)
            chrom_score, chrom_details = cls.compute_chromatic(image_bgr)

            # Combined weighted score
            combined = (
                0.35 * fft_score +
                0.30 * ela_score +
                0.20 * boundary_score +
                0.15 * srm_score
            )
            combined = float(np.clip(combined, 0.0, 1.0))

            return {
                "ela_discrepancy_score": round(float(ela_score), 4),
                "fft_spectral_anomaly": round(float(fft_score), 4),
                "boundary_inconsistency": round(float(boundary_score), 4),
                "combined_score": round(float(combined), 4),
                "details": {
                    "ela": ela_details,
                    "fft": fft_details,
                    "boundary": boundary_details,
                    "srm": srm_details,
                    "chromatic": chrom_details,
                },
            }
        except Exception as exc:
            logger.warning("ImageForensics extract_visual_cues failed: %s", exc)
            return {
                "ela_discrepancy_score": 0.5,
                "fft_spectral_anomaly": 0.5,
                "boundary_inconsistency": 0.5,
                "combined_score": 0.5,
                "details": {"error": str(exc)},
            }

    @classmethod
    def analyze_image_signals(cls, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Backward-compatible wrapper returning complete multi-signal analysis."""
        cues = cls.extract_visual_cues(image_bgr)
        details = cues.get("details", {})

        return {
            "fft_score": cues["fft_spectral_anomaly"],
            "srm_noise_score": round(details.get("srm", {}).get("srm_score", 0.5), 4),
            "ela_score": cues["ela_discrepancy_score"],
            "chromatic_score": round(details.get("chromatic", {}).get("chromatic_score", 0.5), 4),
            "ela_discrepancy_score": cues["ela_discrepancy_score"],
            "fft_spectral_anomaly": cues["fft_spectral_anomaly"],
            "boundary_inconsistency": cues["boundary_inconsistency"],
            "combined_forensic_score": cues["combined_score"],
            "details": details,
        }

    # ── 1. Error Level Analysis (ELA) ──────────────────────────────────────────

    @staticmethod
    def compute_ela(
        image_bgr: np.ndarray,
        quality: int = 90,
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Compute JPEG Error Level Analysis to detect compression gradient anomalies.

        Parameters
        ----------
        image_bgr : np.ndarray
            OpenCV image in BGR format.
        quality : int
            JPEG re-save compression quality level (default: 90).

        Returns
        -------
        Tuple[float, np.ndarray, Dict[str, Any]]
            (ela_score, ela_difference_map, details)
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.5, np.zeros((1, 1, 3), dtype=np.uint8), {"reason": "empty image"}

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Re-save in memory at specified JPEG quality
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved_img = Image.open(buf)

        # Absolute pixel difference
        ela_diff = ImageChops.difference(pil_img, resaved_img)
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale = 255.0 / max(max_diff, 1)
        ela_scaled = ImageEnhance.Brightness(ela_diff).enhance(scale)
        ela_arr = np.array(ela_scaled).astype(np.float32)

        ela_mean = float(np.mean(ela_arr))
        ela_std = float(np.std(ela_arr))

        # Local patch variance across 16x16 grid
        h, w = ela_arr.shape[:2]
        step = 16
        patch_means = []
        for y in range(0, h - step, step):
            for x in range(0, w - step, step):
                patch_means.append(np.mean(ela_arr[y : y + step, x : x + step]))

        patch_variance = float(np.var(patch_means)) if patch_means else 0.0

        # Authentic images have uniform error distribution. Spliced/synthetic images show localized inconsistency.
        inconsistency_ratio = (ela_std + np.sqrt(patch_variance)) / (ela_mean + 1e-4)
        fake_prob = float(1.0 / (1.0 + np.exp(-1.4 * (inconsistency_ratio - 1.25))))
        fake_prob = float(np.clip(fake_prob, 0.05, 0.95))

        return fake_prob, ela_arr.astype(np.uint8), {
            "ela_mean": round(ela_mean, 2),
            "ela_std": round(ela_std, 2),
            "patch_variance": round(patch_variance, 2),
            "inconsistency_ratio": round(inconsistency_ratio, 3),
        }

    # ── 2. 2D FFT Frequency Spectrum Analysis ──────────────────────────────────

    @staticmethod
    def compute_fft_spectrum(image_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Compute 2D FFT and azimuthal radial decay to identify GAN/Diffusion grid spikes.

        Parameters
        ----------
        image_bgr : np.ndarray
            OpenCV image in BGR format.

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (fft_anomaly_score, details)
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.5, {"reason": "empty image"}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        h, w = gray.shape
        if h < 32 or w < 32:
            return 0.5, {"reason": "image too small for 2D FFT"}

        # Resize to power-of-2 standard 256x256
        gray_256 = cv2.resize(gray, (256, 256)).astype(np.float32)

        # 2D Fast Fourier Transform with DC-shift to center
        f_transform = np.fft.fft2(gray_256)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

        # Azimuthal Radial Integration (radial profile from DC center out to Nyquist)
        cy, cx = 128, 128
        y, x = np.ogrid[:256, :256]
        r = np.hypot(x - cx, y - cy).astype(np.int32)

        # Exclude DC component (r < 10)
        r_max = 128
        radial_bins = np.arange(10, r_max)
        radial_power = []
        for rad in radial_bins:
            mask = (r == rad)
            if np.any(mask):
                radial_power.append(float(np.mean(magnitude_spectrum[mask])))
            else:
                radial_power.append(0.0)

        radial_arr = np.array(radial_power, dtype=np.float32)

        # In natural images, power decays smoothly as frequency increases.
        # High-frequency periodic spikes produce high peak-to-mean and abrupt slope jumps.
        high_freq_mask = (x - cx) ** 2 + (y - cy) ** 2 > 20 ** 2
        high_freq_mag = magnitude_spectrum[high_freq_mask]

        mean_val = float(np.mean(high_freq_mag))
        max_val = float(np.max(high_freq_mag))
        std_val = float(np.std(high_freq_mag))

        peak_to_mean = (max_val - mean_val) / (std_val + 1e-6)
        radial_decay_slope = float(radial_arr[0] - radial_arr[-1]) if len(radial_arr) > 1 else 0.0

        # Score computation: abnormal peak-to-mean or unnatural slope indicates GAN/Diffusion upsampling
        fake_prob = float(1.0 / (1.0 + np.exp(-0.85 * (peak_to_mean - 4.75))))
        fake_prob = float(np.clip(fake_prob, 0.05, 0.95))

        return fake_prob, {
            "peak_to_mean_ratio": round(peak_to_mean, 3),
            "high_freq_mean": round(mean_val, 2),
            "high_freq_std": round(std_val, 2),
            "radial_decay_slope": round(radial_decay_slope, 2),
        }

    # ── 3. Boundary Blending & Laplacian Inconsistency ─────────────────────────

    @staticmethod
    def compute_boundary_inconsistency(
        image_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate edge gradient discontinuity and Laplacian noise mismatch across facial borders.

        Parameters
        ----------
        image_bgr : np.ndarray
            Input BGR image array.
        bbox : Tuple[int, int, int, int] | None
            Face bounding box (x1, y1, x2, y2). If None, estimated via center crop.

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (boundary_inconsistency_score, details)
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.5, {"reason": "empty image"}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        h, w = gray.shape

        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            # Fallback center 50% ROI
            cy, cx = h // 2, w // 2
            side = min(h, w) // 2
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            x2, y2 = min(w, cx + side // 2), min(h, cy + side // 2)

        # 1. Laplacian Variance inside vs outside face ROI
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        face_lap = lap[y1:y2, x1:x2]

        inner_var = float(np.var(face_lap)) if face_lap.size > 0 else 20.0

        # Outer perimeter band (8-pixel margin surrounding bounding box)
        margin = 8
        ox1, oy1 = max(0, x1 - margin), max(0, y1 - margin)
        ox2, oy2 = min(w, x2 + margin), min(h, y2 + margin)

        outer_patch = lap[oy1:oy2, ox1:ox2]
        outer_var = float(np.var(outer_patch)) if outer_patch.size > 0 else 20.0

        # Noise discrepancy ratio between inner face and background border
        noise_ratio = abs(inner_var - outer_var) / (max(inner_var, outer_var) + 1e-4)

        # 2. High-Pass Gradient Step at Boundary
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        grad_mag = np.hypot(grad_x, grad_y)

        # Sample border pixels
        border_top = grad_mag[max(0, y1 - 2) : min(h, y1 + 2), x1:x2]
        border_bottom = grad_mag[max(0, y2 - 2) : min(h, y2 + 2), x1:x2]
        border_left = grad_mag[y1:y2, max(0, x1 - 2) : min(w, x1 + 2)]
        border_right = grad_mag[y1:y2, max(0, x2 - 2) : min(w, x2 + 2)]

        border_energies = [
            float(np.mean(p)) for p in (border_top, border_bottom, border_left, border_right)
            if p.size > 0
        ]
        mean_border_step = float(np.mean(border_energies)) if border_energies else 10.0
        face_energy = float(np.mean(grad_mag[y1:y2, x1:x2])) + 1e-4

        boundary_step_ratio = mean_border_step / face_energy

        # Deepfake blending artifacts cause boundary seams (high step ratio) or blurred blending (low step ratio with high noise ratio)
        inconsistency_score = 0.5 * noise_ratio + 0.5 * min(1.0, boundary_step_ratio / 2.0)
        fake_prob = float(1.0 / (1.0 + np.exp(-3.0 * (inconsistency_score - 0.45))))
        fake_prob = float(np.clip(fake_prob, 0.05, 0.95))

        return fake_prob, {
            "inner_laplacian_var": round(inner_var, 2),
            "outer_laplacian_var": round(outer_var, 2),
            "noise_ratio": round(noise_ratio, 3),
            "boundary_step_ratio": round(boundary_step_ratio, 3),
        }

    # ── 4. Spatial Rich Model (SRM) Noise Residuals ───────────────────────────

    @staticmethod
    def compute_srm_noise(image_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Extract sensor noise residual with high-pass spatial filter."""
        if image_bgr is None or image_bgr.size == 0:
            return 0.5, {"srm_score": 0.5}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        residual = cv2.filter2D(gray, -1, kernel)

        h, w = residual.shape
        step = 16
        patch_vars = []
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                patch = residual[i : i + step, j : j + step]
                patch_vars.append(float(np.var(patch)))

        if not patch_vars:
            return 0.5, {"srm_score": 0.5}

        mean_var = float(np.mean(patch_vars))
        std_var = float(np.std(patch_vars))

        if mean_var < 3.0:
            fake_prob = 0.85
        elif mean_var > 90.0:
            fake_prob = 0.75
        else:
            fake_prob = 0.15 + 0.3 * (abs(mean_var - 20.0) / 30.0)

        fake_prob = float(np.clip(fake_prob, 0.05, 0.95))
        return fake_prob, {
            "srm_score": round(fake_prob, 4),
            "noise_mean_variance": round(mean_var, 2),
            "noise_variance_std": round(std_var, 2),
        }

    # ── 5. Chromatic Dispersion ────────────────────────────────────────────────

    @staticmethod
    def compute_chromatic(image_bgr: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Evaluate cross-channel gradient correlation natural to physical camera lenses."""
        if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
            return 0.5, {"chromatic_score": 0.5}

        b, g, r = cv2.split(image_bgr.astype(np.float32))

        grad_r = np.abs(cv2.Sobel(r, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(r, cv2.CV_32F, 0, 1))
        grad_g = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        grad_b = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1))

        rg_diff = float(np.mean(np.abs(grad_r - grad_g)))
        gb_diff = float(np.mean(np.abs(grad_g - grad_b)))
        rb_diff = float(np.mean(np.abs(grad_r - grad_b)))

        total_energy = float(np.mean(grad_g)) + 1e-6
        chrom_metric = float((rg_diff + gb_diff + rb_diff) / (3.0 * total_energy))

        if 0.10 <= chrom_metric <= 0.25:
            fake_prob = 0.15
        elif chrom_metric < 0.07:
            fake_prob = 0.70
        else:
            fake_prob = 0.65

        fake_prob = float(np.clip(fake_prob, 0.05, 0.95))
        return fake_prob, {
            "chromatic_score": round(fake_prob, 4),
            "chromatic_gradient_ratio": round(chrom_metric, 4),
        }
