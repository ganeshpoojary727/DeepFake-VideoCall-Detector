# ⚡ DeepFake Media Detector — REST API Specification

The DeepFake Media Detector provides a FastAPI REST service for single-file and batch media analysis.

## Endpoints

### 1. Detect Single File
- **Route**: `POST /detect/file`
- **Content-Type**: `multipart/form-data`
- **Form Param**: `file` (Binary media file)
- **Response**: `200 OK`
```json
{
  "verdict": "FAKE",
  "confidence": 0.8842,
  "media_type": "video",
  "scores": {
    "video": 0.825,
    "audio": 0.9237,
    "fused": 0.8842
  },
  "processing_time_ms": 1420.5,
  "metadata": {
    "original_filename": "sample.mp4",
    "num_frames": 16,
    "num_faces_detected": 16
  }
}
```

### 2. Detect Batch Files
- **Route**: `POST /detect/batch`
- **Content-Type**: `multipart/form-data`
- **Form Param**: `files` (Array of binary media files)
- **Response**: `200 OK` (Array of `AnalysisReport`)

### 3. System & Model Health
- **Route**: `GET /health`
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "torch_version": "2.11.0+cu128",
  "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "models": {
    "audio_aasist": {
      "checkpoint_exists": true
    },
    "video_efficientnet_transformer": {
      "checkpoint_exists": true
    }
  }
}
```
