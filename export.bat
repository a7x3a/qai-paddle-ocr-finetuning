@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Exporting Kurdish PaddleOCR Checkpoint to Inference Model
echo ==============================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py export %*
) else (
    python main.py export %*
)

if errorlevel 1 (
    echo [ERROR] Export encountered an issue.
    pause
)
endlocal
