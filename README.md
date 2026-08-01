# DeepFake Video Call Detector

**AI-powered real-time deepfake detection for video calls.**

Detects synthetic/cloned audio during live video calls using a CNN trained on Mel spectrogram features from the ASVspoof2019 dataset. Designed for extensibility with planned video deepfake detection and audio-video fusion.

---

## Features

- **Audio Deepfake Detection** — CNN classifier trained on ASVspoof2019 LA
- **Real-Time Microphone Input** — Record and classify live audio
- **File-Based Prediction** — Analyse audio files from disk
- **Video Call Monitoring** — Detect running Zoom/Teams/Meet processes
- **GUI Application** — PyQt6-based desktop interface
- **Production Training Pipeline** — Early stopping, LR scheduling, AMP, TensorBoard, class weighting
- **Three-Way Classification** — REAL / FAKE / UNCERTAIN with configurable thresholds

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | PyTorch 2.0+ |
| Audio Processing | librosa, sounddevice, soundfile |
| Model | DeepFakeCNN (custom lightweight CNN) |
| Features | Mel Spectrogram (128 mels, 2048 n_fft) |
| Dataset | ASVspoof2019 Logical Access |
| Evaluation | scikit-learn + EER metric |
| GUI | PyQt6 |
| Monitoring | psutil |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DeepFake-VideoCall-Detector.git
cd DeepFake-VideoCall-Detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Show available commands
python -m app.main

# Train the model
python -m app.main train

# Run predictions (interactive CLI)
python -m app.main predict

# Evaluate on test set
python -m app.main evaluate

# Launch GUI
python -m app.main gui

# Show project info
python -m app.main info
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
DeepFake-VideoCall-Detector/
├── app/                          # Main application package
│   ├── ai/                       # AI/ML subsystem
│   │   ├── datasets/             # Dataset + DataLoader
│   │   ├── evaluation/           # Model evaluation (+ EER)
│   │   ├── inference/            # Prediction + detection
│   │   ├── models/               # CNN architecture + loader
│   │   ├── preprocessing/        # Audio/video processing
│   │   └── training/             # Training engine
│   ├── config/                   # Dataclass-based configuration
│   ├── core/                     # Domain interfaces & entities
│   ├── gui/                      # PyQt6 GUI
│   ├── monitoring/               # Audio capture, process monitor
│   ├── services/                 # Application services layer
│   ├── tools/                    # Utility scripts
│   └── utils/                    # Logger + helpers
├── docs/                         # Documentation
├── tests/                        # pytest tests
├── trained_models/               # Saved model weights
└── requirements.txt
```

## Architecture

See [docs/Architecture.md](docs/Architecture.md) for the full system architecture.

## Model

The DeepFakeCNN uses 3 convolutional blocks with AdaptiveAvgPool2d for input-size independence:

```
Input: (batch, 1, n_mels, time_frames)
  → Conv2d(1→16) + BN + ReLU + MaxPool + Dropout
  → Conv2d(16→32) + BN + ReLU + MaxPool + Dropout
  → Conv2d(32→64) + BN + ReLU + MaxPool + Dropout
  → AdaptiveAvgPool2d(1,1)
  → Linear(64→128) + ReLU + Dropout
  → Linear(128→2)
Output: (batch, 2)
```

**Parameters**: ~67K (lightweight, fast inference)

## Training

See [docs/Training.md](docs/Training.md) for the training guide.

Key features:
- Early stopping (patience=5)
- CosineAnnealingWarmRestarts scheduler
- Mixed precision (AMP)
- Gradient clipping
- TensorBoard logging
- Class weighting for 1:8.8 imbalance
- Reproducible seeds

## License

This project is for educational and research purposes.

## Acknowledgements

- [ASVspoof2019](https://www.asvspoof.org/) dataset
- PyTorch team
- librosa developers
