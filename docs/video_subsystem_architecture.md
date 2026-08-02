# Video AI Subsystem Architecture & Infrastructure

## Overview

The `app/video` subsystem is a production-grade, highly modular, config-driven framework for video deepfake detection. It mirrors the exact design patterns, SOLID principles, folder philosophy, dependency injection, factory, registry, and builder patterns of the production `app/audio` subsystem.

---

## 1. Folder Philosophy & Purpose

Every subdirectory inside `app/video` serves a single, dedicated architectural responsibility:

| Folder | Architectural Purpose |
|---|---|
| `augmentation/` | Framework for spatial (crop, flip, rotate), temporal (drop, jitter), color, blur, noise, and compression data augmentations. |
| `builders/` | Fluent and config-driven builders (`VideoModelBuilder`, `DatasetBuilder`, `TrainingBuilder`, `AugmentationBuilder`, `VideoPipelineBuilder`). |
| `configs/` | Strongly-typed dataclasses for datasets, models, training, augmentations, and inference. |
| `constants/` | Centralized magic numbers, default resolutions, sequence lengths, normalization constants, and dataset identifiers. |
| `core/` | Abstract base classes (`BaseVideoModel`, `BaseDataset`, `BaseTrainer`, `BaseEvaluator`, `BaseInferenceEngine`, `BaseFeatureExtractor`, `BasePreprocessor`). |
| `datasets/` | Dataset discovery (`DatasetScanner`), metadata caching (`DatasetIndexBuilder` in `datasets/video/cache/`), dataset implementations (`FaceForensicsDataset`, `CelebDFDataset`), and dataloaders (`ProductionVideoDataLoader`). |
| `evaluation/` | Performance & accuracy metrics calculation (`Accuracy`, `Precision`, `Recall`, `F1`, `ROC`, `AUC`, `ConfusionMatrix`, `Latency`, `FPS`, `GPU Memory`). |
| `exceptions/` | Hierarchy of custom domain errors (`VideoError`, `DatasetNotFoundError`, `FrameExtractionError`, `FaceDetectionError`, `ModelInitializationError`, `CheckpointError`, `RegistryError`). |
| `face_alignment/` | Facial landmark alignment interfaces and transformation logic. |
| `face_detection/` | Face detection engines (YuNet DNN detector with skin-segmentation fallback). |
| `features/` | Abstract feature extraction interfaces (`SpatialFeatureExtractor`, `TemporalFeatureExtractor`). |
| `frame_extraction/` | Frame extraction wrappers and video decoding utilities. |
| `inference/` | Real-time streaming inference infrastructure (`WindowCapture`, `FrameQueue`, `VideoDetector`, `InferencePostProcessor`). |
| `models/` | Neural network model wrappers, factory (`VideoFactory`), registry (`ModelRegistry`), and layer abstractions. |
| `pipeline/` | End-to-end training and inference execution pipelines (`InferencePipeline`, `TrainingPipeline`, `ValidationPipeline`). |
| `preprocessing/` | Frame decoding, sampling, face cropping, alignment, normalization, and caching orchestrators (`VideoPreprocessor`). |
| `registry/` | Thread-safe generic component registries (`BaseRegistry`, `ModelRegistry`, `DatasetRegistry`, `OptimizerRegistry`, `SchedulerRegistry`, `AugmentationRegistry`, `LossRegistry`). |
| `training/` | Training execution engine (`ProductionVideoTrainer`), AMP handler, checkpoint manager, early stopping, loss/optimizer/scheduler factories. |
| `utils/` | Device management (`get_device`), logging utilities, and helper functions. |

---

## 2. Core Class Responsibilities

- **`BaseVideoModel`**: Abstract PyTorch `nn.Module` base class enforcing `forward`, `get_num_parameters`, `save`, and `load` contracts.
- **`BaseDataset`**: Abstract PyTorch `Dataset` contract enforcing `__len__`, `__getitem__`, and `get_label_distribution`.
- **`BaseTrainer`**: Abstract contract for training engines.
- **`BaseEvaluator`**: Abstract contract for evaluation engines.
- **`BaseInferenceEngine`**: Abstract contract for real-time frame/sequence inference.
- **`BaseFeatureExtractor`**: Abstract contract for spatial and temporal feature embedding extractors.
- **`DatasetScanner`**: Automatically audits `datasets/video/faceforensics/` and `datasets/video/celebdfv2/` for folder structure integrity, supported categories, and sample counts.
- **`DatasetIndexBuilder`**: Builds, computes statistics for, and caches JSON metadata indexes inside `datasets/video/cache/`.
- **`ProductionVideoDataLoader`**: Multi-worker PyTorch DataLoader factory supporting prefetching, persistent workers, pin memory, and AMP float16/float32 tensor collating.
- **`VideoAugmentationPipeline`**: Composes 9 independently configurable augmentations (crop, flip, color jitter, blur, JPEG compression, noise, rotation, frame dropout, temporal jitter).
- **`ProductionVideoTrainer`**: Full-featured model training engine supporting gradient clipping, AMP float16, TensorBoard logging, early stopping, and top-K checkpoint saving.
- **`InferencePipeline`**: Orchestrates the complete 6-stage video deepfake prediction flow.

---

## 3. End-to-End Pipeline Execution Flow

```
Video Input (File / Stream)
   ↓
[Frame Extraction] -> (FrameExtractor / VideoDecoder)
   ↓
[Face Detection]   -> (FaceDetector / YuNet DNN)
   ↓
[Face Alignment]   -> (FaceAligner / Landmark Transform)
   ↓
[Preprocessing]    -> (FaceCropper / VideoNormalizer)
   ↓
[Feature Extract]  -> (Spatial & Temporal Feature Extractors)
   ↓
[Model Pass]       -> (BaseVideoModel Architecture)
   ↓
[Prediction]       -> (InferencePostProcessor Confidence Output)
```

---

## 4. Future Integration Plans

### EfficientNet-B4 Spatial Integration
Future phases will implement EfficientNet-B4 as the primary spatial backbone. The modular infrastructure is ready for this:
1. `ModelConfig.backbone_name` can be set to `"efficientnet_b4"`.
2. The EfficientNet-B4 feature extractor will subclass `SpatialFeatureExtractor` and register in `ModelRegistry` / `BackboneRegistry`.
3. Preprocessed frames `[B, T, 3, 380, 380]` will pass through the backbone to output spatial feature maps `[B, T, 1792]`.

### Temporal Attention Integration
Future phases will attach temporal attention modules (e.g. Temporal Transformer or Conv-1D Attention) on top of spatial embeddings:
1. Temporal modules will subclass `TemporalFeatureExtractor`.
2. They will receive spatial feature embeddings `[B, T, 1792]` and aggregate frame-to-frame temporal correlations into a sequence embedding `[B, 512]`.
3. The linear classification head will produce final binary logits `[B, 2]` (Real vs Fake).
