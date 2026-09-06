# 🛡️ DeepFake Media Detector

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(350%2F350)-green.svg)](tests/)

**AI-powered static media deepfake detection system for Images, Videos, and Audio.**

Upload any media file to automatically detect synthetic manipulation, facial deepfakes, and cloned/spoofed speech with calibrated confidence scores and detailed modality breakdowns.

---

## 🌟 Key Features

- 🖼️ **Image Deepfake Detection**: Single-frame spatial feature analysis using **EfficientNet-B4** with **YuNet ONNX DNN** face detection and adaptive 20% margin facial cropping.
- 🎬 **Video Deepfake Detection**: Spatiotemporal deepfake detection combining **EfficientNet-B4 + Temporal Transformer** across 16 uniformly sampled frames.
- 🎤 **Audio Deepfake Detection**: Raw waveform audio anti-spoofing powered by **AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*) achieving **99.71% accuracy** and **0.52% Equal Error Rate (EER)** on ASVspoof 2019.
- 🔀 **Multimodal Late Fusion**: Automatic extraction and cross-modal fusion ($0.6\text{Audio} + 0.4\text{Video}$) when analyzing video files with audio tracks.
- 🌐 **Interactive Streamlit Web App**: Clean drag-and-drop UI with media previews, real-time progress bars, and batch export (CSV/JSON).
- ⚡ **High-Performance FastAPI REST Server**: Production-ready asynchronous endpoints for single-file and batch media analysis.
- 💻 **Flexible Command-Line Interface (CLI)**: Fast single-file, batch, and directory scanning directly from your terminal.

---

## 📐 System Architecture

```
                                  ┌─────────────────────────────┐
                                  │      User Input File        │
                                  │   (Image / Video / Audio)   │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │         MediaRouter         │
                                  │   (Format & MIME Analysis)  │
                                  └──────┬───────┬───────┬──────┘
                                         │       │       │
                     ┌───────────────────┘       │       └───────────────────┐
                     │                           │                           │
                     ▼                           ▼                           ▼
          ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
          │    ImageAnalyzer    │     │    VideoAnalyzer    │     │    AudioAnalyzer    │
          ├─────────────────────┤     ├─────────────────────┤     ├─────────────────────┤
          │ • YuNet Face Detect │     │ • Frame Extractor   │     │ • Soundfile Loader  │
          │ • Adaptive Cropper  │     │ • 16-Frame Sampler  │     │ • 16kHz Resampler   │
          │ • 224x224 Normalize │     │ • EfficientNet-B4   │     │ • AASIST Graph Net  │
          │ • EfficientNet-B4   │     │ • Temporal Attention│     │ • Softmax Spoof Prob│
          └──────────┬──────────┘     │ • Audio Extraction  │     └──────────┬──────────┘
                     │                │ • Multimodal Fusion │                │
                     │                └──────────┬──────────┘                │
                     │                           │                           │
                     └───────────────────┐       │       ┌───────────────────┘
                                         │       │       │
                                         ▼       ▼       ▼
                                  ┌─────────────────────────────┐
                                  │       AnalysisReport        │
                                  ├─────────────────────────────┤
                                  │ • Verdict: REAL/FAKE/UNCERT │
                                  │ • Fake Probability (0.0-1.0)│
                                  │ • Per-Modality Breakdown    │
                                  │ • Latency (ms) & Metadata   │
                                  └─────────────────────────────┘
```

---

## 🚀 Quick Start & Deployment

> 📖 **Deploying onto a new laptop?** Check out the step-by-step [**CLONING_AND_SETUP_GUIDE.md**](CLONING_AND_SETUP_GUIDE.md) for full instructions on copying model checkpoints, installing PyTorch CUDA/CPU, and one-click launch scripts.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DeepFake-VideoCall-Detector.git
cd DeepFake-VideoCall-Detector

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Streamlit Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app/ui/streamlit_app.py
# Or via CLI entrypoint:
python -m app.main ui
```

Open your browser at `http://localhost:8501`.

---

### 3. Command-Line Interface (CLI)

```bash
# Analyze a single image, video, or audio file:
python -m app.main predict sample_video.mp4

# Save detection result to JSON:
python -m app.main predict sample_voice.wav -o result.json

# Batch scan an entire folder:
python -m app.main batch path/to/media_folder/ --output batch_report.json

# Check GPU acceleration & model readiness:
python -m app.main health
```

---

### 4. FastAPI REST API Server

Start the REST API daemon:

```bash
# Start server on http://localhost:8000
python -m app.main api --port 8000
# Or directly with Uvicorn:
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

- **Interactive Swagger UI**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **ReDoc Documentation**: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

#### Example cURL Request:

```bash
curl -X POST "http://localhost:8000/detect/file" \
     -H "accept: application/json" \
     -F "file=@test_sample.mp4"
```

#### Example JSON Response:

```json
{
  "verdict": "FAKE",
  "confidence": 0.8842,
  "media_type": "video",
  "scores": {
    "video": 0.8250,
    "audio": 0.9237,
    "fused": 0.8842
  },
  "processing_time_ms": 1420.5,
  "metadata": {
    "original_filename": "test_sample.mp4",
    "analysis_method": "multimodal_fusion",
    "num_frames": 16,
    "num_faces_detected": 16,
    "model": "EfficientNet-B4 + Temporal Transformer"
  }
}
```

---

### 5. Python SDK Usage

```python
from app.analyzer import MediaAnalyzer

# Initialize orchestrator (auto-selects CUDA GPU or CPU)
analyzer = MediaAnalyzer(device="auto")

# 1. Single File Analysis
report = analyzer.analyze("interview_clip.mp4")
print(report.verdict)       # "REAL", "FAKE", or "UNCERTAIN"
print(report.confidence)    # 0.8842
print(report.scores)        # {'video': 0.825, 'audio': 0.9237, 'fused': 0.8842}
print(report.summary)

# 2. Batch Analysis
batch_reports = analyzer.analyze_batch(["photo.jpg", "recording.wav"])
for r in batch_reports:
    print(r.summary)

# 3. Directory Scan
dir_reports = analyzer.analyze_directory("media_folder/")
```

---

## 📊 Model Benchmarks

| Modality | Architecture | Dataset | Accuracy | Metrics | Checkpoint Size |
|---|---|---|---|---|---|
| **Audio** | AASIST (Graph Attention) | ASVspoof 2019 LA | **99.71%** | EER: **0.52%** | 18.1 MB (`trained_models/audio/best_model.pt`) |
| **Video** | EfficientNet-B4 + Temporal Transformer | FaceForensics++ | **69.75%** | AUC: **79.33%**, F1: **73.98%** | 557 MB (`trained_models/video/best_model.pt`) |
| **Image** | EfficientNet-B4 (Single-Frame Mode) | Transfer from FF++ | — | YuNet Cropping + Margin | Shared weights |
| **Multimodal** | Weighted Late Fusion ($0.6A + 0.4V$) | Multi-Modal | — | Threshold: $0.50$ | Zero-parameter |

---

## 🧪 Running Tests

Run the complete test suite:

```bash
# Run all unit and integration tests
pytest -v

# Run media analyzer & API tests
pytest tests/test_media_analyzer.py tests/test_api.py -v

# Run neural network model tests
pytest tests/test_aasist_model.py tests/test_video_model_smoke.py -v
```

---

## 📁 Repository Structure

```
DeepFake-VideoCall-Detector/
├── app/
│   ├── analyzer/            # 🚀 Unified Static Media Analysis Engine
│   │   ├── analysis_report.py   # Standardized dataclass output
│   │   ├── audio_analyzer.py    # AASIST audio detection pipeline
│   │   ├── image_analyzer.py    # EfficientNet-B4 single-frame image pipeline
│   │   ├── media_analyzer.py    # Central orchestrator & batch processor
│   │   ├── media_router.py      # Format detection & file validator
│   │   └── video_analyzer.py    # Spatiotemporal video + audio fusion
│   ├── api/                 # ⚡ FastAPI REST Server
│   │   └── server.py            # /detect/file, /detect/batch, /health endpoints
│   ├── ui/                  # 🌐 Streamlit Web Application
│   │   └── streamlit_app.py     # Multi-tab interactive UI
│   ├── audio/               # 🎤 Audio Anti-Spoofing Subsystem (AASIST)
│   ├── video/               # 🎬 Video Deepfake Subsystem (EfficientNet-B4 + Transformer)
│   ├── fusion/              # 🔀 Multimodal Fusion Engine
│   ├── core/                # 🧩 Core Domain Interfaces
│   ├── config/              # ⚙️ Application Settings
│   ├── utils/               # 🛠️ Logger and Utilities
│   └── main.py              # 💻 Master CLI Entry Point
├── trained_models/          # 🧠 Pre-trained Neural Weights
│   ├── audio/best_model.pt  # Trained AASIST weights
│   └── video/best_model.pt  # Trained Video weights
├── tests/                   # 🧪 Automated Test Suite (350+ tests)
└── requirements.txt
```

---

## ⚖️ Decision Thresholds

- **$P(\text{fake}) \ge 0.70$** $\rightarrow$ **`FAKE`**: High confidence of synthetic manipulation / voice cloning / face swap.
- **$P(\text{fake}) \le 0.30$** $\rightarrow$ **`REAL`**: High confidence authentic organic media.
- **$0.30 < P(\text{fake}) < 0.70$** $\rightarrow$ **`UNCERTAIN`**: Inconclusive signal; recommended for manual review.

---

## 📄 License

This project is licensed under the MIT License for research and educational purposes.
