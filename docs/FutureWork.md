# Future Work

## Short-Term Improvements

### Real-Time Streaming Audio Detection
- Implement sliding window inference on the ring buffer
- Integrate Silero VAD for Voice Activity Detection
- Add exponential moving average confidence tracking
- Add decision hysteresis to prevent label flickering

### GUI Enhancements
- Audio Detection page with live confidence gauge
- Settings page for runtime configuration
- History page with past detection results
- Real-time waveform visualization

### Model Improvements
- Experiment with ECAPA-TDNN architecture (~2-4% EER)
- Add LFCC features alongside Mel spectrogram
- Implement delta + delta-delta features
- Train on ASVspoof2021 for telephony/VoIP robustness

## Medium-Term Improvements

### Video Deepfake Detection
- Integrate MTCNN or RetinaFace for face detection
- Train EfficientNet-B0 or XceptionNet on FaceForensics++
- Implement face alignment with 5-point landmarks
- Frame buffer with temporal consistency checking

### Audio-Video Fusion
- Calibrate fusion weights on validation set
- Implement attention-based fusion (cross-modal attention)
- Add modality-specific confidence calibration (Platt scaling)

### Dataset Expansion
- Add ASVspoof2021 LA (telephony channels)
- Add In-the-Wild dataset (real YouTube deepfakes)
- Simulate video call compression (Opus/AAC codecs)
- Collect real Zoom/Teams samples for evaluation

## Long-Term Vision

### ONNX / TensorRT Deployment
- Export to ONNX for cross-platform inference
- INT8 quantization for 4× size reduction
- TensorRT for 3-5× GPU speedup

### Adversarial Robustness
- Add adversarial training
- Implement input perturbation detection
- Test against known audio adversarial attacks

### Security Hardening
- Model encryption for deployment
- Input sanitization with sandboxed processing
- Rate limiting for API endpoints
- Audit logging for compliance

### Packaging
- PyInstaller single-file executable
- Docker container with GPU support
- CI/CD pipeline with automated testing
- Version pinning with `pip-compile`
