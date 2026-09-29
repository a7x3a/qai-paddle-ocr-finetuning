@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Running Kurdish PaddleOCR Benchmark
echo ==============================================================================

if "%~1"=="" (
    set BENCH_ARGS=--max-samples 100
) else (
    set BENCH_ARGS=%*
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py benchmark-base %BENCH_ARGS%
) else (
    python main.py benchmark-base %BENCH_ARGS%
)

if errorlevel 1 (
    echo [ERROR] Benchmark encountered an issue.
    pause
)
endlocal
