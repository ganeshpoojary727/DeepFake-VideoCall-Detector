# Training Guide

## Quick Start

```bash
python -m app.main train
```

## Configuration

Training parameters are configured via `app/config/settings.py` → `TrainingConfig`:

| Parameter | Default | Env Variable | Description |
|-----------|---------|-------------|-------------|
| `batch_size` | 32 | `BATCH_SIZE` | Mini-batch size |
| `learning_rate` | 0.001 | `LEARNING_RATE` | Initial learning rate |
| `epochs` | 20 | `EPOCHS` | Maximum epochs |
| `num_workers` | 4 | `NUM_WORKERS` | DataLoader worker processes |
| `early_stopping_patience` | 5 | `EARLY_STOPPING_PATIENCE` | Epochs without improvement before stopping |
| `gradient_clip_norm` | 1.0 | `GRADIENT_CLIP_NORM` | Max gradient norm |
| `weight_decay` | 1e-4 | `WEIGHT_DECAY` | AdamW weight decay |
| `seed` | 42 | `SEED` | Random seed for reproducibility |
| `use_mixed_precision` | true | `USE_MIXED_PRECISION` | Enable AMP |

Override via environment variables:
```bash
EPOCHS=50 BATCH_SIZE=64 python -m app.main train
```

## Training Features

### Early Stopping
Monitors validation loss. Stops training after `patience` epochs with no improvement.

### Learning Rate Scheduler
CosineAnnealingWarmRestarts — cyclically decays LR for better convergence.

### Mixed Precision (AMP)
Automatic Mixed Precision on CUDA devices for ~2× speedup.

### Gradient Clipping
Prevents exploding gradients with configurable max norm.

### Class Weighting
ASVspoof2019 has 1:8.8 bonafide-to-spoof ratio. Inverse frequency weights are computed automatically.

### TensorBoard
```bash
tensorboard --logdir runs/
```

### Best Model Saving
Saves the best model based on **validation loss** (not accuracy).

## Dataset

Download ASVspoof2019 LA and place in:
```
app/audio/datasets/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
```
