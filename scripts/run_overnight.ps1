param(
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [switch]$Resume,
    [switch]$FullVal
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  STARTING OVERNIGHT MULTI-DATASET DEEPFAKE TRAINING" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$pythonPath = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "[!] Error: Virtual environment python not found at $pythonPath" -ForegroundColor Red
    exit 1
}

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  - Model:             EfficientNet-B4 + Temporal Transformer"
Write-Host "  - Real Datasets:     FaceForensics++ Original + Celeb-DF Celeb-real + YouTube-real (1890 videos)"
Write-Host "  - Fake Datasets:     FF++ (6 methods) + Celeb-DF Synthesis (11639 videos)"
Write-Host "  - VRAM Optimization: Mixed Precision FP16 (Fits 6GB on RTX 4050)"
Write-Host "  - Sleep Prevention:  Active (Windows will not sleep while training)"
Write-Host "  - Log Directory:     logs\"
Write-Host ""
Write-Host "Training is starting now. You can leave your laptop running overnight." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Run training
$trainArgs = @("--epochs", "$Epochs", "--batch-size", "$BatchSize", "--target-per-class", "1800")
if ($Resume) {
    $latestPath = "trained_models\video\latest.pt"
    if (Test-Path $latestPath) {
        $trainArgs += @("--resume", $latestPath)
        Write-Host "Resuming from checkpoint: $latestPath" -ForegroundColor Yellow
    } else {
        Write-Host "Warning: $latestPath not found, starting fresh" -ForegroundColor Yellow
}
if ($FullVal) {
    $trainArgs += "--full-val"
    Write-Host "Validation Mode: Full (all validation videos every epoch)" -ForegroundColor Yellow
} else {
    Write-Host "Validation Mode: Fast (200 videos ~35s per epoch, full on final epoch)" -ForegroundColor Green
}

& $pythonPath -m app.video.training.train_overnight @trainArgs

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  OVERNIGHT TRAINING FINISHED!" -ForegroundColor Green
Write-Host "  - Model checkpoints in: trained_models\video\" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
