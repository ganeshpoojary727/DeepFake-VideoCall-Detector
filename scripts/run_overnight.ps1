# ============================================================
# 🌙 DeepGuard — Overnight Deepfake Model Training Runner
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  🌙 STARTING OVERNIGHT MULTI-DATASET DEEPFAKE TRAINING" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$pythonPath = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "❌ Error: Virtual environment python not found at $pythonPath" -ForegroundColor Red
    exit 1
}

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$logFile = "logs\overnight_training_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "  • Model:            EfficientNet-B4 + Temporal Transformer"
Write-Host "  • Real Datasets:    FaceForensics++ Original + Celeb-DF Celeb-real + YouTube-real (1,890 videos)"
Write-Host "  • Fake Datasets:    FF++ (6 methods: Deepfakes, FaceShifter, Face2Face, FaceSwap, NeuralTextures, DFD) + Celeb-DF Synthesis (11,639 videos)"
Write-Host "  • VRAM Optimization: Mixed Precision FP16 (Fits within 6GB on RTX 4050)"
Write-Host "  • Sleep Prevention:  Active (Windows will not sleep while training)"
Write-Host "  • Progress Log:     $logFile"
Write-Host ""
Write-Host "🚀 Training is starting now. You can leave your laptop running overnight." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Run training
& $pythonPath -m app.video.training.train_overnight --epochs 20 --batch-size 8 --target-per-class 1800 2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  ✅ OVERNIGHT TRAINING FINISHED!" -ForegroundColor Green
Write-Host "  • Log saved to: $logFile" -ForegroundColor Green
Write-Host "  • Model checkpoints in: trained_models\video\" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
