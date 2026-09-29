@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Starting Kurdish PaddleOCR Production Fine-Tuning
echo ==============================================================================

if "%~1"=="" (
    set TRAIN_ARGS=--epochs 10
) else (
    set TRAIN_ARGS=%*
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py train %TRAIN_ARGS%
) else (
    python main.py train %TRAIN_ARGS%
)

if errorlevel 1 (
    echo [ERROR] Training run encountered an issue.
    pause
)
endlocal
