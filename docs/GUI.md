# GUI Architecture

## Framework

PyQt6 — professional-grade Qt bindings for Python.

## Launch

```bash
python -m app.main gui
```

## Layout

```
┌──────────────────────────────────────────────────┐
│  DeepFake Video Call Detector           ─  □  ×  │
├──────────────────────────────────────────────────┤
│  ┌─────────┐  ┌────────────────────────────────┐ │
│  │ 🏠 Home │  │         Content Area           │ │
│  │ 🎤 Audio│  │  (Stacked pages switched by    │ │
│  │ 📷 Video│  │   sidebar navigation)          │ │
│  │ 📊 Hist │  │                                │ │
│  │ ⚙️ Set  │  │                                │ │
│  └─────────┘  └────────────────────────────────┘ │
├──────────────────────────────────────────────────┤
│  Device: cuda | Model: v2.0 | No video call      │
└──────────────────────────────────────────────────┘
```

## Architecture (MVVM-Ready)

```
View Layer (PyQt6 widgets)
    ↕  data binding
ViewModel Layer (state management)
    ↕  method calls
Service Layer (DetectionService, MonitoringService)
    ↕  method calls
AI Layer (Predictor, Model)
```

## Current Pages

| Page | Status | Description |
|------|--------|-------------|
| Home | ✅ Implemented | Project info, device, video call status |
| Audio Detection | 🔲 Skeleton | Future: real-time audio detection |
| Video Detection | 🔲 Skeleton | Future: video deepfake detection |
| History | 🔲 Skeleton | Future: past detection results |
| Settings | 🔲 Skeleton | Future: configuration editor |

## Extending

To add a new page:
1. Create a `QWidget` subclass
2. Add it to the `QStackedWidget`
3. Add a sidebar entry
