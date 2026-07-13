# AuraBook Training Launcher (PowerShell)
# Run: .\train.ps1

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  AuraBook QLoRA Training" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pyVer = python --version 2>&1
    Write-Host "✅ $pyVer" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check GPU
try {
    $gpuInfo = python -c "import torch; print(f'{torch.cuda.get_device_name(0)}|{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}')" 2>&1
    if ($gpuInfo -match "^(.+)\|(.+)$") {
        $gpuName = $Matches[1]
        $gpuVram = $Matches[2]
        Write-Host "✅ GPU: $gpuName ($gpuVram GB)" -ForegroundColor Green
    } else {
        throw "No GPU"
    }
} catch {
    Write-Host "❌ No GPU detected!" -ForegroundColor Red
    Write-Host "   Install CUDA: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Select training size:" -ForegroundColor Yellow
Write-Host "  1) 50k   (fast, ~30 min)" -ForegroundColor White
Write-Host "  2) 100k  (recommended, ~1 hour)" -ForegroundColor Green
Write-Host "  3) 250k  (~2 hours)" -ForegroundColor White
Write-Host "  4) 500k  (~4 hours)" -ForegroundColor White
Write-Host "  5) 1M    (~8 hours)" -ForegroundColor White
Write-Host "  6) Custom" -ForegroundColor White
Write-Host "  7) Resume checkpoint" -ForegroundColor Yellow
Write-Host "  8) Test model" -ForegroundColor Yellow
Write-Host ""

$choice = Read-Host "Enter choice (1-8)"

switch ($choice) {
    "1" { $SIZE = "50k" }
    "2" { $SIZE = "100k" }
    "3" { $SIZE = "250k" }
    "4" { $SIZE = "500k" }
    "5" { $SIZE = "1M" }
    "6" { $SIZE = Read-Host "Enter size (e.g. 100k)" }
    "7" {
        Write-Host ""
        Write-Host "Resuming from checkpoint..." -ForegroundColor Yellow
        python -m AuraTrainer.scripts.train --resume
        exit 0
    }
    "8" {
        Write-Host ""
        Write-Host "Starting interactive test..." -ForegroundColor Yellow
        python -m AuraTrainer.scripts.inference --model-path ./outputs/merged_model --interactive
        exit 0
    }
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Starting training with $SIZE samples..." -ForegroundColor Green
Write-Host ""

python -m AuraTrainer.scripts.train --size $SIZE

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Training completed!" -ForegroundColor Green
    Write-Host "   Model: .\outputs\" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "   python -m AuraTrainer.scripts.merge --adapter-path .\outputs\lora_adapter"
    Write-Host "   python -m AuraTrainer.scripts.inference --model-path .\outputs\merged_model --interactive"
} else {
    Write-Host ""
    Write-Host "❌ Training failed!" -ForegroundColor Red
}

Write-Host ""
pause
