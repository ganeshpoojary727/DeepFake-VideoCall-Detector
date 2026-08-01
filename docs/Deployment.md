# Deployment Guide

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Train model
python -m app.main train

# Launch GUI
python -m app.main gui
```

## Environment Variables

All training and inference parameters can be overridden:

```bash
# Training
export BATCH_SIZE=64
export LEARNING_RATE=0.0005
export EPOCHS=50
export NUM_WORKERS=8

# Inference
export THRESHOLD_FAKE=0.8
export THRESHOLD_REAL=0.2

# Audio
export SAMPLE_RATE=16000
export N_MELS=128
```

## Production Deployment (Future)

### PyInstaller
```bash
pip install pyinstaller
pyinstaller --onefile --windowed app/main.py
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main", "predict"]
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB |
| GPU | None (CPU works) | NVIDIA with CUDA |
| Storage | 500 MB | 2 GB (with dataset) |
