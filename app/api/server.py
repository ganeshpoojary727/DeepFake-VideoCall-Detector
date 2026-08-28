"""
FastAPI REST API Server for Deepfake Detection with Multimodal Fusion & XAI Diagnostics.

Endpoints
─────────
• GET  /health              — System status, CUDA availability, model readiness
• POST /api/v1/analyze      — Unified Phase 3 Multimodal Ingestion endpoint (ConsolidatedForensicReport)
• POST /detect/file         — Upload a single media file (Image, Video, Audio)
• POST /detect/batch        — Upload multiple media files
• POST /detect/path         — Analyze a media file by server-side file path
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.analyzer.analysis_report import ConsolidatedForensicReport
from app.analyzer.forensic_explainer import ForensicExplainer
from app.analyzer.media_analyzer import MediaAnalyzer
from app.analyzer.media_router import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global lazy analyzer instance
_analyzer: MediaAnalyzer | None = None


def get_analyzer() -> MediaAnalyzer:
    """Return or initialize singleton MediaAnalyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MediaAnalyzer(device="auto")
    return _analyzer


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────


class DiagnosticFactor(BaseModel):
    name: str
    score: int
    status: str
    description: str
    details: str


class ForensicReport(BaseModel):
    threat_level: str
    diagnostic_factors: List[DiagnosticFactor]
    narrative_conclusion: str
    key_indicators: List[str]


class DetectionResponse(BaseModel):
    verdict: str = Field(..., description="REAL, FAKE, or UNCERTAIN")
    confidence: float = Field(..., description="Verdict confidence from 0.0 to 1.0")
    real_confidence: float = Field(default=0.5, description="Calibrated authenticity probability from 0.0 to 1.0")
    fake_confidence: float = Field(default=0.5, description="Calibrated deepfake probability from 0.0 to 1.0")
    media_type: str = Field(..., description="image, video, or audio")
    scores: Dict[str, float | None] = Field(default_factory=dict)
    processing_time_ms: float = Field(..., description="Latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    forensics: Optional[ForensicReport] = None


class PathAnalysisRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative file path to analyze")


class SystemHealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    torch_version: str
    gpu_name: str | None = None
    vram_allocated_mb: float | None = None
    vram_reserved_mb: float | None = None
    models: Dict[str, Any]


class ConsolidatedReportResponse(BaseModel):
    media_type: str = Field(..., description="AUDIO, IMAGE, VIDEO, or MULTIMODAL")
    verdict: str = Field(..., description="REAL or FAKE")
    overall_confidence: float = Field(..., description="Overall confidence in [0.0, 1.0]")
    modality_breakdown: Dict[str, Any] = Field(default_factory=dict)
    temporal_sync: List[Dict[str, Any]] = Field(default_factory=list)
    top_anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    natural_language_report: Optional[Dict[str, Any]] = None
    processing_time_ms: float = Field(..., description="Processing latency in ms")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
# FastAPI App Factory
# ──────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Deepfake Media Detector API",
        description="REST API for static Image, Video, and Audio deepfake detection with XAI forensic diagnostics, Multimodal Fusion, and Generative Reporting.",
        version="4.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for web clients (e.g. Next.js on port 3000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["Info"])
    async def root() -> Dict[str, Any]:
        """API welcome message and documentation link."""
        all_exts = sorted(list(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS))
        return {
            "name": "Deepfake Media Detector API",
            "version": "4.0.0",
            "documentation": "/docs",
            "supported_extensions": all_exts,
            "unified_endpoint": "/api/v1/analyze",
            "export_markdown_endpoint": "/api/v1/export/markdown",
            "export_certificate_endpoint": "/api/v1/export/certificate",
        }

    @app.get("/health", response_model=SystemHealthResponse, tags=["Health"])
    async def health() -> Dict[str, Any]:
        """Check system health, CUDA GPU info, and loaded model checkpoints."""
        analyzer = get_analyzer()
        status_info = analyzer.get_system_status()
        return {
            "status": "healthy",
            **status_info,
        }

    # ── Phase 3 & 4 Unified Multimodal & Export Endpoints ─────────────────────

    @app.post(
        "/api/v1/analyze",
        response_model=ConsolidatedReportResponse,
        status_code=status.HTTP_200_OK,
        tags=["Unified Analysis"],
    )
    async def analyze_file_unified(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Unified endpoint: Ingest Image, Video, or Audio and return ConsolidatedForensicReport with NLP synthesis."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file missing filename")

        ext = Path(file.filename).suffix.lower()
        all_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
        if ext not in all_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed: {sorted(list(all_exts))}",
            )

        analyzer = get_analyzer()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)

        try:
            consolidated: ConsolidatedForensicReport = analyzer.analyze_consolidated(tmp_path)
            consolidated.metadata["original_filename"] = file.filename
            return consolidated.to_dict()
        except Exception as exc:
            logger.exception("API /api/v1/analyze failed for %s: %s", file.filename, exc)
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post(
        "/api/v1/export/markdown",
        status_code=status.HTTP_200_OK,
        tags=["Reporting & Exports"],
    )
    async def export_report_markdown(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Upload media and export a professional GitHub Flavored Markdown audit report."""
        from app.analyzer.forensic_explainer import export_markdown_report

        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file missing filename")

        ext = Path(file.filename).suffix.lower()
        analyzer = get_analyzer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)

        try:
            consolidated: ConsolidatedForensicReport = analyzer.analyze_consolidated(tmp_path)
            md_content = export_markdown_report(consolidated)
            return {
                "file_name": file.filename,
                "verdict": consolidated.verdict,
                "confidence": consolidated.overall_confidence,
                "markdown_report": md_content,
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post(
        "/api/v1/export/certificate",
        status_code=status.HTTP_200_OK,
        tags=["Reporting & Exports"],
    )
    async def export_report_certificate(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Upload media and generate a cryptographically signed ISO/IEC forensic JSON certificate."""
        from app.analyzer.forensic_explainer import export_json_certificate

        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file missing filename")

        ext = Path(file.filename).suffix.lower()
        analyzer = get_analyzer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)

        try:
            consolidated: ConsolidatedForensicReport = analyzer.analyze_consolidated(tmp_path)
            return export_json_certificate(consolidated)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Legacy & Detection Endpoints ───────────────────────────────────────────

    @app.post(
        "/detect/file",
        response_model=DetectionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Detection"],
    )
    async def detect_file(file: UploadFile = File(...)) -> Dict[str, Any]:
        """Upload and analyze a single image, video, or audio file with forensic diagnostics."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file missing filename")

        ext = Path(file.filename).suffix.lower()
        all_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
        if ext not in all_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed: {sorted(list(all_exts))}",
            )

        analyzer = get_analyzer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)

        try:
            report = analyzer.analyze(tmp_path)
            report.metadata["original_filename"] = file.filename
            forensics = ForensicExplainer.explain(report)

            resp = report.to_dict()
            resp["forensics"] = forensics
            return resp
        except Exception as exc:
            logger.exception("API detection failed for %s: %s", file.filename, exc)
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post(
        "/detect/batch",
        response_model=List[DetectionResponse],
        status_code=status.HTTP_200_OK,
        tags=["Detection"],
    )
    async def detect_batch(files: List[UploadFile] = File(...)) -> List[Dict[str, Any]]:
        """Upload and analyze multiple files in a batch."""
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        analyzer = get_analyzer()
        results: List[Dict[str, Any]] = []

        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp_path = Path(tmp.name)
                shutil.copyfileobj(f.file, tmp)

            try:
                report = analyzer.analyze(tmp_path)
                report.metadata["original_filename"] = f.filename
                forensics = ForensicExplainer.explain(report)

                resp = report.to_dict()
                resp["forensics"] = forensics
                results.append(resp)
            except Exception as exc:
                results.append({
                    "verdict": "UNCERTAIN",
                    "confidence": 0.5,
                    "media_type": "unknown",
                    "scores": {},
                    "processing_time_ms": 0.0,
                    "metadata": {"error": str(exc), "original_filename": f.filename},
                    "forensics": None,
                })
            finally:
                tmp_path.unlink(missing_ok=True)

        return results

    @app.post(
        "/detect/path",
        response_model=DetectionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Detection"],
    )
    async def detect_path(request: PathAnalysisRequest) -> Dict[str, Any]:
        """Analyze a media file by server-side file path."""
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        analyzer = get_analyzer()
        try:
            report = analyzer.analyze(file_path)
            forensics = ForensicExplainer.explain(report)
            resp = report.to_dict()
            resp["forensics"] = forensics
            return resp
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return app


# Module-level application instance for uvicorn
app = create_app()
