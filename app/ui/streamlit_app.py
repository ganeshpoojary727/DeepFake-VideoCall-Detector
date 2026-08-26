"""
Deepfake Detector — Production Streamlit Web Application

Multi-Modal Deepfake Detection Interface:
- Tab 1: Single File Analysis (Image, Video, Audio)
- Tab 2: Batch File & Folder Analysis (Multi-upload, table, CSV/JSON export)
- Tab 3: System Health & Model Telemetry (CUDA, VRAM, API docs)

Run with:
    streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.analyzer.media_router import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MediaRouter,
)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Deepfake Media Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header {
        font-size: 32px; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #94a3b8; font-size: 15px; margin-bottom: 20px;
    }
    .verdict-real {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white; padding: 22px; border-radius: 12px;
        text-align: center; font-size: 30px; font-weight: 800;
        margin: 16px 0; box-shadow: 0 6px 16px rgba(16,185,129,0.35);
        letter-spacing: 1px;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white; padding: 22px; border-radius: 12px;
        text-align: center; font-size: 30px; font-weight: 800;
        margin: 16px 0; box-shadow: 0 6px 16px rgba(239,68,68,0.35);
        letter-spacing: 1px;
    }
    .verdict-uncertain {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white; padding: 22px; border-radius: 12px;
        text-align: center; font-size: 30px; font-weight: 800;
        margin: 16px 0; box-shadow: 0 6px 16px rgba(245,158,11,0.35);
        letter-spacing: 1px;
    }
    .metric-box {
        background: #1e1e2e; border: 1px solid #313244;
        border-radius: 10px; padding: 14px; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Cached Analyzer Initialisation
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading deepfake detection neural models…")
def load_analyzer():
    """Load the unified MediaAnalyzer singleton."""
    from app.analyzer.media_analyzer import MediaAnalyzer
    return MediaAnalyzer(device="auto")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=64)
    st.markdown("### 🛡️ Deepfake Detector")
    st.caption("AI-Powered Static Media Verification")
    st.divider()

    st.markdown("**Core Models**")
    st.markdown("• **Audio**: AASIST (ASVspoof 2019, 99.7% acc)")
    st.markdown("• **Video**: EfficientNet-B4 + Temporal Transformer")
    st.markdown("• **Face Detection**: YuNet (ONNX DNN)")
    st.markdown("• **Fusion**: Weighted Late Score ($0.6A + 0.4V$)")

    st.divider()
    analyzer_instance = load_analyzer()
    status_dict = analyzer_instance.get_system_status()

    st.markdown("**Device Status**")
    if status_dict.get("cuda_available"):
        st.success(f"🟢 GPU: {status_dict.get('gpu_name', 'CUDA')}")
        st.caption(f"VRAM: {status_dict.get('vram_allocated_mb', 0)} MB / {status_dict.get('vram_reserved_mb', 0)} MB")
    else:
        st.info("💻 Running on CPU")


# ──────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────

st.markdown('<div class="main-header">🛡️ Deepfake Media Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Analyze images, videos, or audio files for synthetic manipulation and deepfakes.</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Single File Analysis",
    "🗂️ Batch Processing",
    "⚙️ System & API Info",
])


# ──────────────────────────────────────────────
# TAB 1: Single File Analysis
# ──────────────────────────────────────────────

with tab1:
    col_left, col_right = st.columns([1, 1], gap="medium")

    all_extensions = sorted(list(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS))
    allowed_types = [ext.lstrip(".") for ext in all_extensions]

    with col_left:
        st.subheader("1. Upload Media")
        uploaded_file = st.file_uploader(
            "Choose an Image, Video, or Audio file",
            type=allowed_types,
            help=f"Supported formats: {', '.join(allowed_types[:10])}...",
            key="single_uploader",
        )

        if uploaded_file is not None:
            file_ext = Path(uploaded_file.name).suffix.lower()
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            media_type = MediaRouter.detect_type(uploaded_file.name).value.capitalize()

            st.info(f"**File**: `{uploaded_file.name}` • **Type**: {media_type} • **Size**: {file_size_mb:.1f} MB")

            # Media Preview
            if file_ext in IMAGE_EXTENSIONS:
                st.image(uploaded_file, caption="Image Preview", use_container_width=True)
            elif file_ext in VIDEO_EXTENSIONS:
                st.video(uploaded_file)
            elif file_ext in AUDIO_EXTENSIONS:
                st.audio(uploaded_file)

            run_button = st.button("🔎 Run Deepfake Detection", type="primary", use_container_width=True)
        else:
            run_button = False
            st.markdown(
                """
                <div style="border: 2px dashed #334155; border-radius: 12px; padding: 50px 20px; text-align: center; color: #64748b;">
                    <p style="font-size: 40px; margin-bottom: 8px;">📂</p>
                    <p style="font-size: 15px; font-weight: 500;">Drag & drop media file here</p>
                    <p style="font-size: 12px;">Images (.jpg, .png, .webp), Videos (.mp4, .avi, .mov), Audio (.wav, .mp3, .flac)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.subheader("2. Detection Verdict")

        if uploaded_file is not None and run_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)

            try:
                analyzer = load_analyzer()

                with st.spinner("Analyzing neural features…"):
                    report = analyzer.analyze(tmp_path)

                # Verdict Banner
                verdict_styles = {
                    "REAL": ("verdict-real", "✅ REAL / AUTHENTIC"),
                    "FAKE": ("verdict-fake", "❌ DEEPFAKE DETECTED"),
                    "UNCERTAIN": ("verdict-uncertain", "⚠️ UNCERTAIN / INCONCLUSIVE"),
                }
                v_class, v_text = verdict_styles.get(report.verdict, ("verdict-uncertain", report.verdict))

                st.markdown(f'<div class="{v_class}">{v_text}</div>', unsafe_allow_html=True)

                # Confidence Gauge
                st.markdown("#### Fake Probability")
                st.progress(
                    min(report.confidence, 1.0),
                    text=f"Confidence Score: {report.confidence * 100:.1f}%",
                )

                # Per-modality breakdown
                active_scores = {k: v for k, v in report.scores.items() if v is not None}
                if active_scores:
                    st.markdown("#### Modality Breakdown")
                    m_cols = st.columns(len(active_scores))
                    for col, (mod, score) in zip(m_cols, active_scores.items()):
                        with col:
                            color = "🟢" if score <= 0.30 else ("🔴" if score >= 0.70 else "🟡")
                            st.metric(
                                label=f"{color} {mod.capitalize()} Score",
                                value=f"{score * 100:.1f}%",
                            )

                # Summary Statistics
                st.markdown("#### Performance Metrics")
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Latency", f"{report.processing_time_ms:.1f} ms")
                with stat_col2:
                    st.metric("Media Type", report.media_type.capitalize())

                # Metadata Expander
                with st.expander("📋 Detailed Technical Metadata", expanded=False):
                    st.json(report.metadata)

                with st.expander("🗂️ Raw JSON Report", expanded=False):
                    st.json(report.to_dict())

            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                st.exception(exc)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            st.info("Upload a file and click **Run Deepfake Detection** to inspect.")


# ──────────────────────────────────────────────
# TAB 2: Batch Processing
# ──────────────────────────────────────────────

with tab2:
    st.subheader("Batch File Verification")
    st.markdown("Upload multiple media files to analyze them simultaneously and export structured reports.")

    batch_files = st.file_uploader(
        "Upload multiple files",
        type=allowed_types,
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.markdown(f"**Selected {len(batch_files)} file(s)**")

        if st.button("🚀 Process Batch Files", type="primary"):
            analyzer = load_analyzer()
            progress_bar = st.progress(0, text="Starting batch analysis…")

            results_list = []
            for idx, b_file in enumerate(batch_files):
                b_ext = Path(b_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=b_ext) as tmp:
                    tmp.write(b_file.getvalue())
                    t_path = Path(tmp.name)

                try:
                    rep = analyzer.analyze(t_path)
                    res_entry = {
                        "File Name": b_file.name,
                        "Verdict": rep.verdict,
                        "Fake Probability (%)": round(rep.confidence * 100, 2),
                        "Media Type": rep.media_type,
                        "Latency (ms)": round(rep.processing_time_ms, 1),
                    }
                    for mod, sc in rep.scores.items():
                        if sc is not None:
                            res_entry[f"{mod.capitalize()} Score (%)"] = round(sc * 100, 2)
                    results_list.append(res_entry)
                except Exception as exc:
                    results_list.append({
                        "File Name": b_file.name,
                        "Verdict": "ERROR",
                        "Fake Probability (%)": None,
                        "Media Type": "Unknown",
                        "Latency (ms)": 0.0,
                    })
                finally:
                    t_path.unlink(missing_ok=True)

                progress_bar.progress(
                    (idx + 1) / len(batch_files),
                    text=f"Processed {idx + 1} of {len(batch_files)}: {b_file.name}",
                )

            df = pd.DataFrame(results_list)

            # Summary Counters
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Processed", len(df))
            with c2:
                fakes_count = (df["Verdict"] == "FAKE").sum()
                st.metric("🔴 Fake Count", fakes_count)
            with c3:
                real_count = (df["Verdict"] == "REAL").sum()
                st.metric("🟢 Real Count", real_count)
            with c4:
                unc_count = (df["Verdict"] == "UNCERTAIN").sum()
                st.metric("🟡 Uncertain Count", unc_count)

            # Results Table
            st.dataframe(df, use_container_width=True)

            # Export options
            csv_data = df.to_csv(index=False).encode("utf-8")
            json_data = json.dumps(results_list, indent=2).encode("utf-8")

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    "📥 Download CSV Report",
                    data=csv_data,
                    file_name="deepfake_batch_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with d_col2:
                st.download_button(
                    "📥 Download JSON Report",
                    data=json_data,
                    file_name="deepfake_batch_report.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ──────────────────────────────────────────────
# TAB 3: System & API Info
# ──────────────────────────────────────────────

with tab3:
    st.subheader("System Telemetry & REST API Integration")

    telemetry_col1, telemetry_col2 = st.columns(2)

    with telemetry_col1:
        st.markdown("#### Hardware & Environment")
        st.json(analyzer_instance.get_system_status())

    with telemetry_col2:
        st.markdown("#### FastAPI REST API Integration")
        st.markdown("The detector includes a built-in FastAPI server for programmatic access.")
        st.code(
            """# 1. Start the API server
uvicorn app.api:app --host 0.0.0.0 --port 8000

# 2. Inspect Interactive Swagger Docs
http://localhost:8000/docs

# 3. Detect via cURL
curl -X POST "http://localhost:8000/detect/file" \\
     -H "accept: application/json" \\
     -F "file=@sample_video.mp4"
""",
            language="bash",
        )

    st.divider()
    st.markdown(
        """
        ### 📖 Classification Logic
        - **$P(\\text{fake}) \\ge 0.70$** $\\rightarrow$ **`FAKE`**: Strong acoustic or spatiotemporal deepfake artifacts detected.
        - **$P(\\text{fake}) \\le 0.30$** $\\rightarrow$ **`REAL`**: High confidence authentic natural media.
        - **$0.30 < P(\\text{fake}) < 0.70$** $\\rightarrow$ **`UNCERTAIN`**: Ambiguous signal requiring further manual review.
        """
    )
