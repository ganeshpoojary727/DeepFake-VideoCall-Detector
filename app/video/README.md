# Video AI Subsystem Infrastructure

`app/video` provides the complete modular infrastructure for video deepfake detection, mirroring the architecture of `app/audio`.

## Key Infrastructure Components

- **Configuration (`app/video/configs`)**: Config-driven dataclasses for datasets, models, training, augmentations, and inference.
- **Constants (`app/video/constants`)**: Centralized hyperparameters, default resolutions, sequence lengths, normalization constants, and dataset identifiers (`faceforensics`, `celebdfv2`).
- **Exceptions (`app/video/exceptions`)**: Custom domain error hierarchy.
- **Registries (`app/video/registry`)**: Thread-safe component registries (`ModelRegistry`, `DatasetRegistry`, `OptimizerRegistry`, `SchedulerRegistry`, `AugmentationRegistry`, `LossRegistry`).
- **Builders (`app/video/builders`)**: Fluent builders (`VideoModelBuilder`, `DatasetBuilder`, `TrainingBuilder`, `AugmentationBuilder`, `VideoPipelineBuilder`).
- **Base Classes (`app/video/core`)**: Abstract interfaces (`BaseVideoModel`, `BaseDataset`, `BaseTrainer`, `BaseEvaluator`, `BaseInferenceEngine`, `BaseFeatureExtractor`).
- **Dataset Infrastructure (`app/video/datasets`)**: `DatasetScanner`, `DatasetIndexBuilder` (cached in `datasets/video/cache/`), `ProductionVideoDataLoader`.
- **Augmentation Framework (`app/video/augmentation`)**: 9 independently configurable spatial, temporal, color, blur, noise, and compression augmentations.
- **Preprocessing Framework (`app/video/preprocessing`)**: Frame extraction, decoding, face detection, cropping, alignment, normalization, and caching.
- **Training Engine (`app/video/training`)**: Multi-epoch trainer with AMP float16, gradient clipping, early stopping, and checkpointing.
- **Evaluation Engine (`app/video/evaluation`)**: Accuracy, Precision, Recall, F1, ROC, AUC, Confusion Matrix, Latency, FPS, GPU memory metrics.
- **Inference Infrastructure (`app/video/inference`)**: `WindowCapture`, `FrameQueue`, `VideoDetector`, `InferencePostProcessor`.
- **Pipeline (`app/video/pipeline`)**: Stage-by-stage `InferencePipeline` and `TrainingPipeline`.

## Quick Start Example

```python
from app.video.configs import DatasetConfig, ModelConfig, VideoTrainingConfig
from app.video.builders import VideoPipelineBuilder

# Build training pipeline
builder = VideoPipelineBuilder()
pipeline = (
    builder
    .with_model_config(ModelConfig(model_name="video_detector"))
    .with_training_config(VideoTrainingConfig(epochs=10))
    .with_dataset_config(DatasetConfig(dataset_name="faceforensics"))
    .build_training_pipeline()
)
```
