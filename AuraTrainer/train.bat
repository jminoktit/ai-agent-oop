@echo off
chcp 65001 >nul 2>&1
title AuraBook Training Launcher

echo ═══════════════════════════════════════════════════════════
echo   AuraBook Training Launcher
echo ═══════════════════════════════════════════════════════════
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Check GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" >nul 2>&1
if errorlevel 1 (
    echo ❌ No GPU detected!
    echo    Install CUDA: https://developer.nvidia.com/cuda-downloads
    echo    Install cuDNN: https://developer.nvidia.com/cudnn
    pause
    exit /b 1
)

:: Show GPU info
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)')"
echo.

:: Menu
echo Choose training size:
echo   1) 50k   samples (fast, ~30 min)
echo   2) 100k  samples (recommended, ~1 hour)
echo   3) 250k  samples (~2 hours)
echo   4) 500k  samples (~4 hours)
echo   5) 1M    samples (~8 hours)
echo   6) Custom
echo   7) Resume from checkpoint
echo   8) Test model only
echo.
set /p choice="Enter choice (1-8): "

if "%choice%"=="1" set SIZE=50k
if "%choice%"=="2" set SIZE=100k
if "%choice%"=="3" set SIZE=250k
if "%choice%"=="4" set SIZE=500k
if "%choice%"=="5" set SIZE=1M
if "%choice%"=="6" (
    set /p SIZE="Enter size (e.g. 100k): "
)
if "%choice%"=="7" (
    python -m AuraTrainer.scripts.train --resume
    pause
    exit /b 0
)
if "%choice%"=="8" (
    python -m AuraTrainer.scripts.inference --model-path ./outputs/merged_model --interactive
    pause
    exit /b 0
)

echo.
echo Starting training with %SIZE% samples...
echo.

python -m AuraTrainer.scripts.train --size %SIZE%

if errorlevel 1 (
    echo.
    echo ❌ Training failed! Check logs for details.
) else (
    echo.
    echo ✅ Training completed!
    echo    Model saved to: ./outputs/
)

pause
