@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Running Kurdish PaddleOCR Pre-Flight Smoke Test
echo ==============================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py smoke-test %*
) else (
    python main.py smoke-test %*
)

if errorlevel 1 (
    echo [ERROR] Smoke test encountered an issue.
    pause
)
endlocal
