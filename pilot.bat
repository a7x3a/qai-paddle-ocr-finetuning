@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Running Kurdish PaddleOCR Rapid Pilot Convergence Test
echo ==============================================================================

if "%~1"=="" (
    set PILOT_ARGS=--num-samples 500
) else (
    set PILOT_ARGS=%*
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py pilot-run %PILOT_ARGS%
) else (
    python main.py pilot-run %PILOT_ARGS%
)

if errorlevel 1 (
    echo [ERROR] Pilot run encountered an issue.
    pause
)
endlocal
