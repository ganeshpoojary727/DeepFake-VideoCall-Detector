# 🚀 Complete Cloning & Local Deployment Guide
### Running the DeepFake Video Call Detector & Web Platform on Another Laptop

> **Target Audience:** Developers, Researchers, or Evaluators deploying this repository onto a fresh or secondary machine.  
> **Supported OS:** Windows 10/11 (Primary), Linux (Ubuntu 20.04+), macOS (CPU mode).

---

## 📋 Table of Contents
1. [Crucial Notice: Model Weights & Git](#1-crucial-notice-model-weights--git)
2. [Hardware & Software Prerequisites](#2-hardware--software-prerequisites)
3. [Step 1: Clone the Repository](#3-step-1-clone-the-repository)
4. [Step 2: Transfer and Place Model Checkpoints](#4-step-2-transfer-and-place-model-checkpoints)
5. [Step 3: Setup Python Environment & PyTorch](#5-step-3-setup-python-environment--pytorch)
6. [Step 4: Setup Next.js Web Frontend](#6-step-4-setup-nextjs-web-frontend)
7. [Step 5: Run the Backend & Frontend](#7-step-5-run-the-backend--frontend)
8. [Step 6: One-Click Startup Scripts (Optional)](#8-step-6-one-click-startup-scripts-optional)
9. [Step 7: Verification & Smoke Testing](#9-step-7-verification--smoke-testing)
10. [Troubleshooting & Common Issues](#10-troubleshooting--common-issues)

---

## 1. Crucial Notice: Model Weights & Git

> [!WARNING]
> **Trained model weights (`*.pt`, `*.pth`) are excluded by `.gitignore` due to GitHub's 100 MB file limit.**  
> If you run `git clone` on a new laptop, **the code will clone, but the neural network weights will NOT be there.**  
> You must manually copy the `trained_models/` folder from your current machine to the new machine.

### Exact Model Checkpoints Required:
```
DeepFake-VideoCall-Detector/
└── trained_models/
    ├── audio/
    │   └── best_model.pt          (~18.2 MB — AASIST Audio Spoof Model)
    └── video/
        ├── best_model.pt          (~687 MB — EfficientNet-B4 + Temporal Transformer)
        └── (optional) best_auc.pt (~687 MB — Best AUC checkpoint)
```

---

## 2. Hardware & Software Prerequisites

### A. Hardware
* **RAM:** 8 GB minimum (16 GB recommended).
* **GPU:**
  * **NVIDIA GPU (Recommended):** GTX 1660, RTX 2060, RTX 3050, RTX 4050 or higher with 4 GB+ VRAM.
  * **CPU Only (Supported):** The system automatically detects CUDA and falls back to CPU if no NVIDIA GPU is found. Inference will just take 3–5 seconds longer per video.
* **Disk Space:** ~5 GB free space (including PyTorch, node_modules, and model checkpoints).

### B. Software
1. **Git**: [git-scm.com](https://git-scm.com/)
2. **Python**: **Python 3.10, 3.11, 3.12, or 3.13** (64-bit).  
   * *Windows install tip:* Check the box **"Add python.exe to PATH"** during installation.
3. **Node.js**: **v18.0.0 or v20+ (LTS)** with `npm`: [nodejs.org](https://nodejs.org/)
4. **FFmpeg** (Required for demuxing audio tracks from video containers):
   * **Windows:** Install via winget: `winget install Gyan.FFmpeg` or download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) and add its `bin` folder to your System PATH.
   * **Linux:** `sudo apt install ffmpeg`
   * **macOS:** `brew install ffmpeg`

---

## 3. Step 1: Clone the Repository

Open a terminal (PowerShell, Command Prompt, or Bash) on your new laptop:

```bash
git clone <YOUR_REPOSITORY_URL>
cd DeepFake-VideoCall-Detector
```

---

## 4. Step 2: Transfer and Place Model Checkpoints

On your **current laptop**, prepare the model checkpoints:

### Method A: Zip & Transfer via USB Drive or Cloud (Recommended)
1. Compress the `trained_models` folder:
   * Right-click `DeepFake-VideoCall-Detector/trained_models` → **Compress to ZIP file**.
2. Copy `trained_models.zip` to the new laptop via USB drive, Google Drive, OneDrive, or local network.
3. On the **new laptop**, extract `trained_models` into the root of `DeepFake-VideoCall-Detector/`.

### Ensure this exact folder hierarchy exists on the new laptop:
```
DeepFake-VideoCall-Detector/
├── trained_models/
│   ├── audio/
│   │   └── best_model.pt
│   └── video/
│       ├── best_model.pt
│       └── best_auc.pt
```

*(Note: The face detector `models_cache/face_detection_yunet_2023mar.onnx` will download automatically from OpenCV Zoo on first run if not already present).*

---

## 5. Step 3: Setup Python Environment & PyTorch

In the project root directory:

### 1. Create a Virtual Environment
**Windows (PowerShell or CMD):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
*(If PowerShell shows an `Execution_Policies` script restriction error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run `.venv\Scripts\activate` again).*

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Install PyTorch with CUDA (or CPU)

#### If your new laptop has an NVIDIA GPU:
Check your CUDA version by running `nvidia-smi` in terminal.
* For **CUDA 12.1 / 12.4 / 12.8**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
* For **CUDA 11.8**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

#### If your new laptop does NOT have an NVIDIA GPU (CPU mode):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### 3. Install Required Dependencies
With the virtual environment still activated:
```bash
pip install -r requirements.txt
```

Verify the installation:
```bash
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```
* If CUDA is available, it will print `CUDA Available: True`.  
* If it prints `False`, it will seamlessly use CPU without breaking anything.

---

## 6. Step 4: Setup Next.js Web Frontend

Open a **new terminal** window (or navigate to `frontend`):

```bash
cd frontend
npm install
```

This installs all required React 19, Next.js 15/16, Tailwind CSS, Framer Motion, and Lucide icon packages.

To test that the frontend builds without errors:
```bash
npm run build
```

---

## 7. Step 5: Run the Backend & Frontend

To run the complete platform, you will run **two servers** simultaneously:

### Terminal 1: FastAPI Backend (Port 8000)
Navigate to the root directory and activate the virtual environment:

**Windows:**
```powershell
cd DeepFake-VideoCall-Detector
.venv\Scripts\activate
python -m uvicorn app.api.server:create_app --factory --host 127.0.0.1 --port 8000 --reload
```

**Linux / macOS:**
```bash
cd DeepFake-VideoCall-Detector
source .venv/bin/activate
python -m uvicorn app.api.server:create_app --factory --host 127.0.0.1 --port 8000 --reload
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### Terminal 2: Next.js Frontend (Port 3000)
Open a second terminal window:

```bash
cd DeepFake-VideoCall-Detector/frontend
npm run dev
```

You should see:
```
▲ Next.js
- Local:   http://localhost:3000
- Network: http://<your-ip>:3000
✓ Ready in 600ms
```

---

### Open Your Browser
Navigate to:
👉 **`http://localhost:3000`**

You now have the full deepfake detection platform running locally with:
- Drag-and-drop media upload for video, audio, and images
- Dynamic Dual-Stream Breakdown (**Video Fake % vs Audio Spoof %**)
- Adversarial Gating telemetry
- Classical forensic cues (ELA, 2D FFT, Boundary Laplacian)
- Interactive PDF/JSON report exports

---

## 8. Step 6: One-Click Startup Scripts (Optional)

To avoid manually typing terminal commands every time on Windows, create these two convenience scripts:

### File 1: `start_backend.bat` (Place in project root)
```bat
@echo off
title DeepFake Detector - FastAPI Backend
echo Starting FastAPI Backend on http://127.0.0.1:8000...
call .venv\Scripts\activate
python -m uvicorn app.api.server:create_app --factory --host 127.0.0.1 --port 8000
pause
```

### File 2: `start_frontend.bat` (Place in project root)
```bat
@echo off
title DeepFake Detector - Next.js Frontend
echo Starting Next.js Frontend on http://localhost:3000...
cd frontend
npm run dev
pause
```

Now, simply double-click `start_backend.bat` and `start_frontend.bat` whenever you want to launch the system!

---

## 9. Step 7: Verification & Smoke Testing

### 1. Verify Backend Health
Open your browser or run:
```bash
curl http://127.0.0.1:8000/health
```
**Expected JSON Response:**
```json
{
  "status": "healthy",
  "device": "cuda" or "cpu",
  "cuda_available": true,
  "torch_version": "2.x.x",
  "models": {
    "audio_aasist": { "checkpoint_exists": true },
    "video_efficientnet_transformer": { "checkpoint_exists": true }
  }
}
```
Ensure both `checkpoint_exists` are `true`!

### 2. Verify Interactive Swagger API Docs
Open: **`http://127.0.0.1:8000/docs`**  
Test endpoints like `POST /api/v1/analyze` directly from the browser.

### 3. Verify Media Analysis from UI
1. Go to `http://localhost:3000`.
2. Upload any test video (`.mp4`), audio (`.wav`/`.mp3`), or image (`.jpg`/`.png`).
3. Click **"Analyze Media"**.
4. Check that:
   - Verdict shows `REAL` or `FAKE` with confidence percentage.
   - **Modality Breakdown Card** renders Video % and Audio % separately.
   - Evidence Matrix displays all 5 forensic indicators without `NaN` or `undefined`.
   - Markdown and JSON export buttons work.

---

## 10. Troubleshooting & Common Issues

### Issue 1: `checkpoint_exists: false` or "Weights failed to load"
* **Cause:** The checkpoint `.pt` files were not copied over from the original machine.
* **Fix:** Check that `trained_models/audio/best_model.pt` and `trained_models/video/best_model.pt` exist and are larger than 15 MB and 500 MB respectively.

### Issue 2: `RuntimeError: CUDA out of memory`
* **Cause:** Your secondary laptop has limited VRAM (e.g. 4GB or integrated graphics) and another app is using GPU memory.
* **Fix:** You can force CPU execution without changing code:
  * Set environment variable in terminal before starting backend:
    ```powershell
    $env:CUDA_VISIBLE_DEVICES = "-1"
    python -m uvicorn app.api.server:create_app --factory --host 127.0.0.1 --port 8000
    ```

### Issue 3: `FileNotFoundError: ffmpeg not found` when processing videos
* **Cause:** `librosa` cannot demux the audio track from the video container without FFmpeg.
* **Fix:** Install FFmpeg via `winget install Gyan.FFmpeg` on Windows, or install via system package manager on Linux/Mac, and restart your terminal.

### Issue 4: `About: Execution_Policies` error in PowerShell
* **Cause:** Windows restricts running PowerShell scripts by default.
* **Fix:** Run this once in PowerShell:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```

### Issue 5: Port 8000 or 3000 is already in use
* **Cause:** A previous instance of uvicorn or node is still running in the background.
* **Fix on Windows:**
  ```powershell
  # Find and kill process on port 8000
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  # Find and kill process on port 3000
  Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process -Force
  ```

### Issue 6: YuNet ONNX Download Failed (`urllib.error.URLError`)
* **Cause:** Firewall or offline network preventing auto-download of `face_detection_yunet_2023mar.onnx`.
* **Fix:** Manually copy `models_cache/face_detection_yunet_2023mar.onnx` from the original laptop into the `models_cache/` directory on the new laptop.

---

## 🚀 Summary Checklist for the New Laptop
- [ ] Cloned repo with Git
- [ ] Transferred and placed `trained_models/audio/best_model.pt` and `trained_models/video/best_model.pt`
- [ ] Created Python `.venv` and activated it
- [ ] Installed PyTorch (CUDA or CPU) + `pip install -r requirements.txt`
- [ ] Ran `cd frontend && npm install`
- [ ] Verified `http://127.0.0.1:8000/health` shows `checkpoint_exists: true`
- [ ] Tested full analysis on `http://localhost:3000`
