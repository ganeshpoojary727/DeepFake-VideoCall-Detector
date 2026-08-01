# Inference Guide

## File Prediction

```bash
python -m app.main predict
# Choose option 1, enter audio file path
```

## Microphone Recording

```bash
python -m app.main predict
# Choose option 2, speak for 5 seconds
```

## Programmatic Usage

```python
from app.audio.models.model_loader import ModelLoader
from app.audio.inference.predictor import Predictor
from app.config.settings import settings

# Load model
loader = ModelLoader()
model = loader.load()

# Create predictor
predictor = Predictor(model=model, device=settings.DEVICE)

# Predict
result = predictor.predict("path/to/audio.wav")
print(f"Label: {result.label.value}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Latency: {result.latency_ms:.1f} ms")
```

## Three-Way Classification

| Score Range | Label | Meaning |
|-------------|-------|---------|
| P(spoof) ≥ 0.7 | FAKE | High confidence deepfake |
| P(spoof) ≤ 0.3 | REAL | High confidence authentic |
| 0.3 < P(spoof) < 0.7 | UNCERTAIN | Requires further analysis |

Thresholds are configurable via `THRESHOLD_FAKE` and `THRESHOLD_REAL` environment variables.

## Supported Audio Formats

`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, `.aac`

Maximum file size: 50 MB
