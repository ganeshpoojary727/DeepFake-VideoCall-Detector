# Folder Structure

```
DeepFake-VideoCall-Detector/
│
├── app/                              # Main application package
│   ├── __init__.py                   # Package init
│   ├── main.py                       # CLI entry point (train/predict/evaluate/gui)
│   │
│   ├── audio/                        # Audio AI subsystem
│   │   ├── datasets/                 # AudioDataset, DataLoader factories
│   │   ├── preprocessing/            # AudioPreprocessor (load, normalize, trim)
│   │   ├── augmentation/             # GaussianNoise, VolumePerturbation, SpecAugment
│   │   ├── features/                 # FeatureExtractor (Mel spectrograms)
│   │   ├── models/                   # DeepFakeCNN, LightCNN, ModelLoader, ModelRegistry
│   │   ├── training/                 # Trainer, train script
│   │   ├── inference/                # Predictor, VoiceDetector, StreamingAudioDetector, predict script
│   │   ├── evaluation/               # Evaluator, test script
│   │   └── utils/
│   │
│   ├── video/                        # Video AI subsystem
│   │   ├── datasets/                 # VideoDataset
│   │   ├── preprocessing/            # VideoPreprocessor
│   │   ├── augmentation/
│   │   ├── frame_extraction/
│   │   ├── face_detection/           # FaceDetector, FaceBox
│   │   ├── face_alignment/
│   │   ├── models/                   # VideoDeepFakeCNN
│   │   ├── training/                 # train_video script
│   │   ├── inference/                # VideoDetector
│   │   ├── evaluation/
│   │   └── utils/
│   │
│   ├── fusion/                       # Audio-Video Multimodal Fusion subsystem
│   │   ├── models/
│   │   ├── training/
│   │   ├── inference/                # FusionEngine, PredictionManager
│   │   ├── evaluation/
│   │   └── utils/
│   │
│   ├── config/
│   │   └── settings.py               # Dataclass config with env overrides
│   │
│   ├── core/
│   │   └── interfaces.py             # ABCs, dataclasses, enums (domain layer)
│   │
│   ├── gui/
│   │   ├── main_window.py            # PyQt6 main window
│   │   └── widgets/                  # GUI components
│   │
│   ├── monitoring/
│   │   ├── audio_capture.py          # Ring buffer audio capture
│   │   ├── process_monitor.py        # Video call process detection
│   │   └── screen_capture.py         # Screen capture skeleton
│   │
│   ├── services/
│   │   ├── detection_service.py      # Detection orchestration service
│   │   ├── camera_service.py
│   │   ├── microphone_service.py
│   │   ├── monitoring_service.py     # System monitoring service
│   │   └── event_bus.py
│   │
│   ├── tools/
│   │   └── visualize_spectrogram.py  # Spectrogram visualization utility
│   │
│   └── utils/
│       ├── logger.py                 # Rotating file + console logging
│       └── helpers.py                # Validation, temp files, seeds, hashing
│
├── docs/                             # Documentation
├── tests/                            # pytest test suite
├── trained_models/                   # Saved model weights (gitignored)
├── logs/                             # Application logs (gitignored)
├── .gitignore
├── requirements.txt
├── deepfake_project_audit.md         # Architecture audit
└── README.md
```
