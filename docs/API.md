# API Reference

## Core Interfaces (`app.core.interfaces`)

### `PredictionResult`
```python
@dataclass
class PredictionResult:
    label: DetectionLabel      # REAL, FAKE, UNCERTAIN
    confidence: float          # 0.0 – 1.0
    modality: Modality         # AUDIO, VIDEO, FUSED
    latency_ms: float          # inference time
    model_version: str
```

### `EvaluationResult`
```python
@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float | None          # Equal Error Rate
    confusion_matrix: ndarray
    classification_report: str
```

### `DetectionLabel` (Enum)
`REAL`, `FAKE`, `UNCERTAIN`

### `Modality` (Enum)
`AUDIO`, `VIDEO`, `FUSED`

---

## Models (`app.audio.models`)

### `DeepFakeCNN(num_classes=2)`
Lightweight CNN for Mel-spectrogram classification.

### `ModelLoader(model_path, device, expected_hash)`
- `.load(warmup=True) → DeepFakeCNN`

---

## Preprocessing (`app.audio.preprocessing`)

### `AudioPreprocessor(sample_rate=16000)`
- `.preprocess(audio_path) → (ndarray, int)`
- `.load_audio(path) → (ndarray, int)`
- `.normalize_audio(audio) → ndarray`
- `.trim_silence(audio) → ndarray`

### `FeatureExtractor(apply_augmentation=False)`
- `.extract(audio) → Tensor`
- `.create_mel_spectrogram(audio) → ndarray`
- `.normalize_spectrogram(mel_db) → ndarray`
- `.spec_augment(tensor) → Tensor`

---

## Inference (`app.audio.inference`)

### `Predictor(model, device, threshold_fake, threshold_real)`
- `.predict(audio_path) → PredictionResult`

### `VoiceDetector(predictor)`
- `.detect(duration=5) → PredictionResult`

### `PredictionManager(audio_predictor, video_detector)`
- `.predict_audio(path) → PredictionResult`
- `.predict_fused(audio_path, video_input) → PredictionResult`

---

## Services (`app.services`)

### `DetectionService(on_result=None)`
- `.initialise(model_path=None)`
- `.detect_file(audio_path) → PredictionResult`
- `.detect_fused(audio_path, video_input) → PredictionResult`

### `MonitoringService()`
- `.check_video_calls() → VideoCallStatus`

---

## Training (`app.audio.training`)

### `Trainer(model, train_loader, validation_loader, optimizer, criterion, device, ...)`
- `.fit(epochs, save_path) → None`
- `.train_one_epoch() → EpochMetrics`
- `.validate() → EpochMetrics`
- `.save_checkpoint(epoch, path)`
- `.load_checkpoint(path) → int`

---

## Configuration (`app.config.settings`)

### `settings` (module-level singleton)
- `.audio` → `AudioConfig`
- `.training` → `TrainingConfig`
- `.model` → `ModelConfig`
- `.inference` → `InferenceConfig`
- `.DEVICE` → `torch.device` (lazy)
- `.validate() → list[str]`
