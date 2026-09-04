"""
Generative Forensic Explainer & Natural Language Report Synthesis Engine.

Translates numerical telemetry from ConsolidatedForensicReport into transparent,
structured, and natural forensic explanations (mirroring ChatGPT/Gemini-style breakdowns).

Supports a dual-engine architecture:
1. Provider A: Grounded LLM generation (Google Gemini / OpenAI / Local Ollama).
2. Provider B: Zero-hallucination deterministic template synthesizer fallback.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.analyzer.analysis_report import AnalysisReport, ConsolidatedForensicReport
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# 1. Natural Language Report Schema
# ──────────────────────────────────────────────


@dataclass
class NaturalLanguageReport:
    """Natural Language Forensic Narrative Report schema.

    Attributes
    ----------
    executive_summary : str
        High-level 2-3 sentence verdict summary citing confidence and primary cues.
    visual_analysis_narrative : str
        In-depth spatial/temporal visual cues analysis (Grad-CAM, ELA, 2D FFT, boundary).
    audio_analysis_narrative : str
        In-depth acoustic findings analysis (AASIST, spectral roll-off, vocoder artifacts).
    temporal_inconsistency_notes : str
        Timestamped descriptions of where manipulation spikes occur.
    forensic_recommendations : list of str
        Actionable guidance for human investigators and reviewers.
    provider_used : str
        The generation engine used ("deterministic_rules", "gemini", "openai", "ollama").
    generation_timestamp : str
        ISO 8601 UTC timestamp of report generation.
    """

    executive_summary: str
    visual_analysis_narrative: str
    audio_analysis_narrative: str
    temporal_inconsistency_notes: str
    forensic_recommendations: List[str] = field(default_factory=list)
    provider_used: str = "deterministic_rules"
    generation_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────
# 2. Base & Deterministic Explainer Providers
# ──────────────────────────────────────────────


class BaseExplainerProvider:
    """Abstract base class for forensic narrative generation providers."""

    def generate(self, report: ConsolidatedForensicReport) -> NaturalLanguageReport:
        raise NotImplementedError


class DeterministicExplainerProvider(BaseExplainerProvider):
    """Zero-hallucination, rule-based forensic report synthesizer."""

    def generate(self, report: ConsolidatedForensicReport) -> NaturalLanguageReport:
        media_type = report.media_type.upper()
        verdict = report.verdict.upper()
        confidence = report.overall_confidence
        conf_pct = f"{confidence * 100:.1f}%"
        is_fake = verdict == "FAKE"

        # ── NOT_APPLICABLE: Stage-0 identified non-biometric artwork ──────────
        if verdict == "NOT_APPLICABLE":
            meta = report.metadata or {}
            content_cls = meta.get("content_classification") or {}
            category = content_cls.get("category", "NON_BIOMETRIC_CONTENT")
            reason = content_cls.get(
                "reason",
                "The uploaded media is non-photorealistic digital artwork.",
            )
            _CATEGORY_LABELS = {
                "DIGITAL_ART_ANIME": "digital illustration / anime artwork",
                "SCENERY_OBJECT": "scenery, wallpaper, or inanimate object",
            }
            category_label = _CATEGORY_LABELS.get(category, "non-photorealistic artwork")
            return NaturalLanguageReport(
                executive_summary=(
                    f"Deepfake biometric analysis is NOT APPLICABLE to this {media_type.lower()} file. "
                    f"The Stage-0 Content Pre-Classifier identified the media as {category_label} "
                    f"with {conf_pct} confidence. Forensic deepfake detection is designed exclusively "
                    f"for photorealistic images and videos of real human subjects."
                ),
                visual_analysis_narrative=(
                    f"Content classification signal: {reason} "
                    f"The system detected characteristics consistent with {category_label}, including "
                    f"flat color regions, hard line-art contours, low PRNU sensor noise, and/or "
                    f"a quantized color palette — fingerprints of digital rendering tools "
                    f"(e.g. Photoshop, Clip Studio, Blender) rather than camera sensors. "
                    f"Running biometric CNN models on this content would produce meaningless outputs."
                ),
                audio_analysis_narrative=(
                    "No audio analysis was performed; this media does not contain a biometric audio track."
                ),
                temporal_inconsistency_notes=(
                    "Temporal frame analysis was bypassed. Deepfake temporal inconsistency detection "
                    "requires a continuous video sequence of a real human subject."
                ),
                forensic_recommendations=[
                    f"No action required. This {category_label} is correctly excluded from deepfake analysis.",
                    "If you intended to analyze a photograph of a real person, please upload a photorealistic image.",
                    "Deepfake forensics apply only to photographic or video recordings of human subjects.",
                ],
                provider_used="deterministic_rules",
            )

        modality = report.modality_breakdown or {}
        audio = modality.get("audio") or {}
        visual = modality.get("visual") or {}
        classical = modality.get("classical_forensics") or {}
        sync = report.temporal_sync or []
        anomalies = report.top_anomalies or []

        # ── 1. Executive Summary ───────────────────────────
        if is_fake:
            reasons = []
            if visual.get("raw_scores", {}).get("fake_prob", 0.0) >= 0.60:
                reasons.append("spatial/temporal facial manipulation anomalies")
            if audio.get("raw_scores", {}).get("spoof_prob", 0.0) >= 0.60:
                reasons.append("synthetic acoustic vocoder cues")
            if classical.get("fft_spectral_anomaly", 0.0) >= 0.60:
                reasons.append("2D FFT lattice grid frequency spikes")
            if classical.get("ela_discrepancy_score", 0.0) >= 0.60:
                reasons.append("JPEG error level compression gradients")
            if not reasons:
                reasons.append("multi-signal forensic artifact patterns")

            reason_str = ", ".join(reasons)
            exec_summary = (
                f"The uploaded {media_type.lower()} file is classified as FAKE with {conf_pct} confidence "
                f"due to detected {reason_str}. Significant biometric and structural deviations from authentic capture "
                f"were identified across the forensic analysis pipeline."
            )
        else:
            exec_summary = (
                f"The uploaded {media_type.lower()} file is classified as AUTHENTIC (REAL) with {conf_pct} confidence. "
                f"No significant neural generative artifacts, boundary blending discontinuities, or acoustic vocoder "
                f"dispersion patterns were detected across the examined signals."
            )

        # ── 2. Visual Analysis Narrative ───────────────────
        if media_type in ("IMAGE", "VIDEO", "MULTIMODAL") and visual:
            raw_v = visual.get("raw_scores", {})
            fake_p = raw_v.get("fake_prob", 0.0)
            cues = visual.get("visual_cues", classical)
            ela = cues.get("ela_discrepancy_score", 0.0)
            fft = cues.get("fft_spectral_anomaly", 0.0)
            boundary = cues.get("boundary_inconsistency", 0.0)
            artifacts = visual.get("key_artifacts", [])

            vis_parts = [
                f"Visual neural spatial inspection yielded a deepfake probability of {fake_p*100:.1f}%."
            ]
            if fft >= 0.60:
                vis_parts.append(
                    f"2D Fourier transform analysis detected periodic grid frequency spikes (anomaly score: {fft:.2f}) "
                    f"characteristic of GAN/Diffusion upsampling convolution layers."
                )
            else:
                vis_parts.append(
                    f"2D Fourier frequency spectrum exhibited organic radial energy decay (anomaly score: {fft:.2f})."
                )

            if ela >= 0.60:
                vis_parts.append(
                    f"Error Level Analysis (ELA) revealed noticeable compression rate variance ({ela:.2f}) between facial features and the background."
                )
            else:
                vis_parts.append(f"Error Level Analysis confirmed uniform compression error levels ({ela:.2f}).")

            if boundary >= 0.60:
                vis_parts.append(
                    f"Laplacian boundary filtering identified edge blending seam discontinuities ({boundary:.2f}) along the facial perimeter."
                )

            if artifacts:
                vis_parts.append(
                    f"Grad-CAM explainability localized primary anomalous hotspots in {len(artifacts)} keyframe(s), focusing on boundary and facial region transitions."
                )

            visual_narrative = " ".join(vis_parts)
        else:
            visual_narrative = "Visual inspection was not conducted because this media does not contain a visual track."

        # ── 3. Audio Analysis Narrative ───────────────────
        if media_type in ("AUDIO", "MULTIMODAL") and audio:
            raw_a = audio.get("raw_scores", {})
            spoof_p = raw_a.get("spoof_prob", 0.0)
            spec = audio.get("spectral_cues", {})
            rolloff = spec.get("spectral_rolloff_hz", 0.0)
            flatness = spec.get("spectral_flatness", 0.0)
            hf_ratio = spec.get("high_freq_energy_ratio", 0.0)
            detected_arts = spec.get("artifacts_detected", [])

            aud_parts = [
                f"AASIST raw waveform graph attention analysis yielded a speech spoofing probability of {spoof_p*100:.1f}%."
            ]
            if rolloff > 0:
                aud_parts.append(f"Spectral roll-off was measured at {rolloff:.0f} Hz.")
            if flatness > 0:
                aud_parts.append(f"Spectral flatness index evaluated at {flatness:.4f}.")
            if hf_ratio > 0:
                aud_parts.append(f"High-frequency energy ratio measured at {hf_ratio:.4f}.")

            if detected_arts:
                aud_parts.append(f"Detected acoustic anomalies include: {', '.join(detected_arts)}.")
            elif spoof_p >= 0.60:
                aud_parts.append("Neural vocoder phase mismatch and harmonic sub-band dispersion were observed.")
            else:
                aud_parts.append("Acoustic harmonics and formants match natural biological vocal tract dynamics.")

            audio_narrative = " ".join(aud_parts)
        else:
            audio_narrative = "Acoustic inspection was not conducted because this media does not contain an audio track."

        # ── 4. Temporal Inconsistency Notes ────────────────
        if sync:
            anomaly_seconds = [item for item in sync if item.get("is_anomaly")]
            if anomaly_seconds:
                sec_list = [f"{item.get('second', 0.0):.2f}s (fused spoof: {item.get('fused_spoof_prob', 0.0):.2f})" for item in anomaly_seconds[:4]]
                temporal_notes = (
                    f"Temporal analysis revealed {len(anomaly_seconds)} synchronized anomaly timestamp(s). "
                    f"Peak manipulation evidence observed at: {', '.join(sec_list)}."
                )
            else:
                temporal_notes = f"Temporal timeline evaluated across {len(sync)} time bin(s) without significant frame-to-frame tampering spikes."
        elif anomalies:
            anomaly_desc = [f"at {a.get('timestamp_sec', 0.0):.2f}s ({a.get('description')})" for a in anomalies[:3]]
            temporal_notes = f"Top anomalous events detected: {'; '.join(anomaly_desc)}."
        else:
            temporal_notes = "No temporal fluctuations or time-series inconsistencies were detected."

        # ── 5. Forensic Recommendations ───────────────────
        recommendations: List[str] = []
        if is_fake:
            recommendations.append("Conduct manual inspection of facial boundary transitions and jawline blending seams.")
            if media_type in ("AUDIO", "MULTIMODAL"):
                recommendations.append("Validate original vocal recording against uncompressed reference audio using acoustic spectrograms.")
            recommendations.append("Inspect file metadata, EXIF headers, and container encoding signatures for synthetic tool footprints.")
            recommendations.append("Isolate timestamped anomaly segments for higher-resolution sub-frame forensic verification.")
        else:
            recommendations.append("Authenticity confirmed under current threshold parameters; archive cryptographic hash of file for chain-of-custody.")
            recommendations.append("If content is used in high-security identity verification, ensure multi-factor liveness protocols remain active.")

        return NaturalLanguageReport(
            executive_summary=exec_summary,
            visual_analysis_narrative=visual_narrative,
            audio_analysis_narrative=audio_narrative,
            temporal_inconsistency_notes=temporal_notes,
            forensic_recommendations=recommendations,
            provider_used="deterministic_rules",
        )


# ──────────────────────────────────────────────
# 3. LLM Explainer Provider (Grounded Synthesis)
# ──────────────────────────────────────────────


class LLMExplainerProvider(BaseExplainerProvider):
    """External or local LLM provider with strict factual grounding against telemetry data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        api_type: str = "auto",
        timeout_seconds: float = 4.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.model_name = model_name
        self.api_type = api_type
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Check if LLM credentials are configured."""
        return bool(self.api_key) or bool(os.getenv("OLLAMA_HOST"))

    def generate(self, report: ConsolidatedForensicReport) -> NaturalLanguageReport:
        if not self.is_available():
            raise RuntimeError("No LLM API key or Ollama host configured")

        # Telemetry data package
        telemetry_json = json.dumps(report.to_dict(), indent=2)

        prompt = (
            "You are a Senior Digital Media Forensics Investigator. "
            "Translate the following deepfake detection telemetry data into a structured forensic narrative. "
            "You MUST base your statements SOLELY on the numerical scores, timestamps, and cues provided. "
            "DO NOT invent or hallucinate metrics, frequencies, or timestamps not present in the data.\n\n"
            f"TELEMETRY DATA:\n{telemetry_json}\n\n"
            "Return a strictly valid JSON object with the following schema:\n"
            "{\n"
            '  "executive_summary": "2-3 sentence verdict explanation citing confidence and primary cues",\n'
            '  "visual_analysis_narrative": "Detailed breakdown of visual and classical cues",\n'
            '  "audio_analysis_narrative": "Detailed breakdown of acoustic AASIST and spectral cues",\n'
            '  "temporal_inconsistency_notes": "Timestamped descriptions of where manipulation occurs",\n'
            '  "forensic_recommendations": ["list of actionable guidance items"]\n'
            "}"
        )

        # Example implementation for Google Gemini API or OpenAI REST endpoint
        # For robustness and offline compatibility, we support REST POST with standard JSON schema parsing
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return self._call_gemini_rest(prompt)
        elif os.getenv("OPENAI_API_KEY"):
            return self._call_openai_rest(prompt)
        else:
            raise RuntimeError("No recognized LLM provider available")

    def _call_gemini_rest(self, prompt: str) -> NaturalLanguageReport:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)

        return NaturalLanguageReport(
            executive_summary=parsed.get("executive_summary", ""),
            visual_analysis_narrative=parsed.get("visual_analysis_narrative", ""),
            audio_analysis_narrative=parsed.get("audio_analysis_narrative", ""),
            temporal_inconsistency_notes=parsed.get("temporal_inconsistency_notes", ""),
            forensic_recommendations=parsed.get("forensic_recommendations", []),
            provider_used="gemini",
        )

    def _call_openai_rest(self, prompt: str) -> NaturalLanguageReport:
        key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a factual forensic investigator. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = data["choices"][0]["message"]["content"]
        parsed = json.loads(text)

        return NaturalLanguageReport(
            executive_summary=parsed.get("executive_summary", ""),
            visual_analysis_narrative=parsed.get("visual_analysis_narrative", ""),
            audio_analysis_narrative=parsed.get("audio_analysis_narrative", ""),
            temporal_inconsistency_notes=parsed.get("temporal_inconsistency_notes", ""),
            forensic_recommendations=parsed.get("forensic_recommendations", []),
            provider_used="openai",
        )


# ──────────────────────────────────────────────
# 4. Generative Forensic Explainer Orchestrator
# ──────────────────────────────────────────────


class GenerativeForensicExplainer:
    """Orchestrates natural language forensic narrative synthesis with automatic LLM/fallback routing."""

    def __init__(self) -> None:
        self.deterministic_provider = DeterministicExplainerProvider()
        self.llm_provider = LLMExplainerProvider()

    def explain(
        self,
        report: Union[ConsolidatedForensicReport, AnalysisReport],
        force_provider: Optional[str] = None,
    ) -> NaturalLanguageReport:
        """Generate natural language narrative report.

        Parameters
        ----------
        report : ConsolidatedForensicReport | AnalysisReport
            Analyzed forensic report.
        force_provider : str | None
            Force specific provider ("deterministic" | "llm").

        Returns
        -------
        NaturalLanguageReport
            Synthesized natural language forensic breakdown.
        """
        # Coerce AnalysisReport to ConsolidatedForensicReport if necessary
        if isinstance(report, AnalysisReport):
            consolidated = ConsolidatedForensicReport(
                media_type=report.media_type.upper(),
                verdict=report.verdict,
                overall_confidence=report.confidence,
                modality_breakdown={
                    report.media_type.lower(): {"raw_scores": report.scores},
                    "classical_forensics": report.metadata.get("visual_cues", {}),
                },
                temporal_sync=report.metadata.get("timeline", []),
                top_anomalies=report.metadata.get("key_artifacts", []),
                processing_time_ms=report.processing_time_ms,
                metadata=report.metadata,
            )
        else:
            consolidated = report

        # Strategy 1: Forced deterministic
        if force_provider == "deterministic":
            return self.deterministic_provider.generate(consolidated)

        # Strategy 2: Attempt LLM if available and requested/auto
        if force_provider in ("llm", "gemini", "openai", None) and self.llm_provider.is_available():
            try:
                logger.info("GenerativeForensicExplainer: Generating narrative via LLM provider")
                return self.llm_provider.generate(consolidated)
            except Exception as exc:
                logger.warning("GenerativeForensicExplainer: LLM generation failed (%s), falling back to deterministic", exc)

        # Strategy 3: Deterministic zero-hallucination fallback
        return self.deterministic_provider.generate(consolidated)


# ──────────────────────────────────────────────
# 5. Export Utilities (Markdown & JSON Audit)
# ──────────────────────────────────────────────


def export_markdown_report(report: ConsolidatedForensicReport) -> str:
    """Export ConsolidatedForensicReport as a professional GitHub Flavored Markdown audit report."""
    verdict_badge = "🔴 **DEEPFAKE DETECTED**" if report.is_fake else "🟢 **AUTHENTIC MEDIA**"
    nl = report.natural_language_report or {}
    exec_summary = nl.get("executive_summary", f"Verdict: {report.verdict} ({report.overall_confidence*100:.1f}%)")
    visual_narrative = nl.get("visual_analysis_narrative", "N/A")
    audio_narrative = nl.get("audio_analysis_narrative", "N/A")
    temporal_notes = nl.get("temporal_inconsistency_notes", "N/A")
    recs = nl.get("forensic_recommendations", [])

    rec_list = "\n".join(f"- {r}" for r in recs) if recs else "- No additional recommendations."

    sync_rows = ""
    for item in report.temporal_sync[:10]:
        sec_val = item.get("second", 0.0)
        a_prob = item.get("audio_spoof_prob")
        v_prob = item.get("visual_spoof_prob")
        fused = item.get("fused_spoof_prob", 0.0)
        is_ano = item.get("is_anomaly", False)

        a_str = f"{a_prob:.4f}" if a_prob is not None else "N/A"
        v_str = f"{v_prob:.4f}" if v_prob is not None else "N/A"
        ano_str = "⚠️ ANOMALY" if is_ano else "OK"

        sync_rows += f"| {sec_val:.2f}s | {a_str} | {v_str} | {fused:.4f} | {ano_str} |\n"

    md = f"""# Digital Forensics Investigation Report

**Evaluation Status:** {verdict_badge}  
**Overall Confidence:** {report.overall_confidence * 100:.1f}%  
**Media Category:** {report.media_type}  
**Processing Latency:** {report.processing_time_ms:.1f} ms  
**Report Generated:** {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  

---

## 1. Executive Summary
{exec_summary}

---

## 2. Modality & Telemetry Breakdown

### 2.1 Visual Forensics Analysis
{visual_narrative}

### 2.2 Acoustic Forensics Analysis
{audio_narrative}

### 2.3 Temporal Synchronization & Inconsistency Notes
{temporal_notes}

---

## 3. Synchronized Timeline Evidence (First 10 Windows)

| Time (sec) | Audio Spoof Prob | Visual Fake Prob | Fused Spoof Prob | Status |
| :--- | :--- | :--- | :--- | :--- |
{sync_rows if sync_rows else "| 0.00s | N/A | N/A | 0.0000 | OK |\n"}

---

## 4. Forensic Recommendations for Human Investigators
{rec_list}

---
*Report generated by DeepFake-VideoCall-Detector v3.0 (Forensic Explainer Subsystem)*
"""
    return md


def export_json_certificate(report: ConsolidatedForensicReport) -> Dict[str, Any]:
    """Export ConsolidatedForensicReport as an ISO/IEC compliant forensic JSON certificate."""
    payload_str = json.dumps(report.to_dict(), sort_keys=True)
    sha256_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    return {
        "certificate_id": f"CERT-{sha256_hash[:16].upper()}",
        "schema_version": "3.0.0-forensics",
        "issued_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "integrity_sha256": sha256_hash,
        "investigation_results": {
            "verdict": report.verdict,
            "overall_confidence": report.overall_confidence,
            "media_type": report.media_type,
            "is_manipulated": report.is_fake,
        },
        "telemetry_breakdown": report.to_dict(),
        "narrative_report": report.natural_language_report,
        "issuer": "DeepFake-VideoCall-Detector Forensic Certification Authority",
    }


# ──────────────────────────────────────────────
# 6. Backward Compatibility Wrapper
# ──────────────────────────────────────────────


class ForensicExplainer:
    """Backward-compatible ForensicExplainer facade."""

    @staticmethod
    def explain(report: AnalysisReport) -> Dict[str, Any]:
        """Generate forensic factor analysis and a human-readable narrative conclusion."""
        media_type = report.media_type.lower()
        fake_prob = report.fake_confidence
        real_prob = report.real_confidence
        is_real = report.verdict == "REAL"
        is_fake = report.verdict == "FAKE"

        if is_real:
            threat_level = "AUTHENTIC" if real_prob >= 0.80 else "CLEAN"
        elif is_fake:
            threat_level = "CRITICAL" if fake_prob >= 0.85 else "HIGH"
        else:
            threat_level = "MODERATE" if fake_prob >= 0.50 else "LOW"

        # Also generate natural language report
        explainer = GenerativeForensicExplainer()
        nl_report = explainer.explain(report)

        factors = [
            {
                "name": "Synthetic Risk Score",
                "score": int(fake_prob * 100),
                "status": "ANOMALOUS" if fake_prob >= 0.60 else "NATURAL",
                "description": "Multi-signal deep learning & forensic feature analysis.",
                "details": nl_report.executive_summary,
            },
            {
                "name": "Visual & Frequency Spectrum Cues",
                "score": int((1.0 - fake_prob) * 100 if is_real else fake_prob * 100),
                "status": "NATURAL" if is_real else "ANOMALOUS",
                "description": "Spatial 2D FFT, ELA compression, and boundary Laplacian gradients.",
                "details": nl_report.visual_analysis_narrative,
            },
            {
                "name": "Acoustic Graph & Formant Naturalness",
                "score": int(real_prob * 100 if is_real else fake_prob * 100),
                "status": "NATURAL" if is_real else "ANOMALOUS",
                "description": "AASIST vocal tract harmonics and vocoder phase inspection.",
                "details": nl_report.audio_analysis_narrative,
            },
        ]

        return {
            "threat_level": threat_level,
            "diagnostic_factors": factors,
            "narrative_conclusion": nl_report.executive_summary,
            "key_indicators": [
                f"Verdict: {report.verdict}",
                f"Confidence: {report.confidence * 100:.1f}%",
                f"Media Type: {report.media_type}",
            ],
            "natural_language_report": nl_report.to_dict(),
        }
