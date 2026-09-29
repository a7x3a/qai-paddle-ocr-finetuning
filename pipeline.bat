@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Launching End-to-End Kurdish PaddleOCR Automated Pipeline
echo  (Smoke -^> Baseline -^> Pilot -^> Train -^> Benchmark -^> Export)
echo ==============================================================================

if "%~1"=="" (
    set PIPE_ARGS=--epochs 10 --no-interactive
) else (
    set PIPE_ARGS=%*
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py pipeline %PIPE_ARGS%
) else (
    python main.py pipeline %PIPE_ARGS%
)

if errorlevel 1 (
    echo [ERROR] Pipeline encountered an issue.
    pause
)
endlocal
