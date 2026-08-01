# Project Roadmap

## ✅ Phase 1: Foundation Fixes (Complete)
- [x] Fix `LOG_DIR` bug in Settings
- [x] Remove dead `networkx` import
- [x] Fix `video_detector.py` copy-paste
- [x] Add `AdaptiveAvgPool2d` to CNN
- [x] Add spectrogram normalization
- [x] Fix inconsistent indentation (all files)
- [x] Create core interfaces (ABCs, dataclasses, enums)
- [x] Dataclass-based Settings with env var overrides
- [x] Centralized logging with rotation

## ✅ Phase 2: Training Improvements (Complete)
- [x] Add `num_workers=4` to DataLoaders
- [x] Add class weights to CrossEntropyLoss
- [x] Add learning rate scheduler (CosineAnnealing)
- [x] Add early stopping (patience=5)
- [x] Add SpecAugment augmentation
- [x] Add EER metric to Evaluator
- [x] Add feature caching support
- [x] Add `torch.manual_seed()` for reproducibility
- [x] Add TensorBoard logging
- [x] Add mixed precision training
- [x] Add gradient clipping
- [x] Add checkpoint save/resume

## ✅ Phase 3: GUI Implementation (Complete)
- [x] PyQt6 main window with navigation
- [x] Dashboard view with device/model info
- [x] Video call detection status
- [x] MVVM-ready architecture

## 🔲 Phase 4: Real-Time Audio Detection (Future)
- [ ] Streaming audio capture with ring buffer (skeleton implemented)
- [ ] Voice Activity Detection (Silero VAD)
- [ ] Sliding window inference
- [ ] Confidence tracking (exponential moving average)
- [ ] Decision engine with hysteresis
- [ ] Thread-safe GUI updates

## 🔲 Phase 5: Video Detection (Future)
- [ ] Face detection (MTCNN)
- [ ] Face alignment
- [ ] Video deepfake model (EfficientNet/XceptionNet)
- [ ] Webcam capture integration

## 🔲 Phase 6: Fusion (Future)
- [x] Late fusion engine (skeleton implemented)
- [ ] Weighted confidence combination testing
- [ ] Three-way decision calibration

## 🔲 Phase 7: Optimization (Future)
- [ ] ONNX export
- [ ] Quantization (INT8)
- [ ] TorchScript compilation
- [ ] Performance benchmarking

## 🔲 Phase 8: Deployment (Future)
- [ ] PyInstaller packaging
- [ ] Docker container
- [ ] CI/CD pipeline
