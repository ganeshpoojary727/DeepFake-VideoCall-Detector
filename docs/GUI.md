# 🌐 DeepFake Media Detector — User Interfaces

The DeepFake Media Detector provides two frontend interfaces:

## 1. Streamlit Web App (Built-in Python UI)

### Launch Command
```bash
python -m app.main ui
# Or directly:
streamlit run app/ui/streamlit_app.py
```

### Features
- **Single File Analyzer**: Drag & drop Image, Video, or Audio file with instant media playback and confidence gauge.
- **Batch Scanner**: Upload multiple files simultaneously, view live progress, inspect summary statistics, and export reports to CSV or JSON.
- **Hardware Telemetry**: Real-time CUDA GPU name, VRAM allocation, and model status.

---

## 2. Next.js + Framer Motion Frontend (Modern Web Client)

A standalone React/Next.js frontend communicating via the FastAPI REST API (`http://localhost:8000`).

### Architecture
- **Framework**: Next.js App Router + TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **Animation**: Framer Motion (animated scanning laser, radial progress, spring verdict cards)
- **API**: Communicates with `POST http://localhost:8000/detect/file`
