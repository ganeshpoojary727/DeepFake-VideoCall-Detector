# System Architecture

## Overview

The DeepFake Video Call Detector follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│                 Presentation Layer                    │
│              (GUI / CLI / main.py)                    │
├─────────────────────────────────────────────────────┤
│                  Service Layer                       │
│     (DetectionService, MonitoringService)             │
├─────────────────────────────────────────────────────┤
│                    AI Layer                          │
│  (Models, Preprocessing, Training, Inference)        │
├─────────────────────────────────────────────────────┤
│                  Domain Layer                        │
│   (Interfaces, Entities, Enums — core/)              │
├─────────────────────────────────────────────────────┤
│               Infrastructure Layer                   │
│          (Config, Logger, Helpers)                    │
└─────────────────────────────────────────────────────┘
```

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Factory** | `ModelLoader`, DataLoader factories | Create complex objects with configuration |
| **Strategy** | `BaseFeatureExtractor` hierarchy | Swap mel/MFCC/LFCC extractors |
| **Template Method** | `BaseDetector.detect()` | Common detection flow |
| **Singleton** | `settings` module-level instance | Single configuration source |
| **Dependency Injection** | All services accept dependencies | Testability |
| **Observer** | `DetectionService.on_result` callback | Decouple detection from UI |

## Data Flow

### Training Pipeline
```
ASVspoof2019 FLAC → AudioPreprocessor (load → trim → normalize)
    → FeatureExtractor (mel → dB → normalize → resize → tensor)
    → AudioDataset (protocol parsing + caching)
    → DataLoader (batching + workers)
    → Trainer (train + validate + early stop + checkpoint)
    → best_model.pth
```

### Inference Pipeline
```
Audio Input (file or mic) → AudioPreprocessor → FeatureExtractor
    → DeepFakeCNN (forward pass) → Softmax → Three-way decision
    → PredictionResult (REAL / FAKE / UNCERTAIN)
```

## Module Dependencies

```
core/interfaces.py       ← NO dependencies (domain layer)
config/settings.py       ← pathlib, os (infrastructure)
utils/logger.py          ← config
utils/helpers.py         ← config, logger
ai/models/cnn_model.py   ← torch (no project deps)
ai/preprocessing/        ← config, core, logger
ai/datasets/             ← preprocessing, config, logger
ai/training/             ← config, logger
ai/evaluation/           ← core, logger
ai/inference/            ← preprocessing, core, config, helpers, logger
services/                ← ai/inference, ai/models, config, logger
gui/                     ← services, config, logger
main.py                  ← all (entry point)
```
