@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py infer %*
) else (
    python main.py infer %*
)

if errorlevel 1 (
    echo [ERROR] Inference exited with an error.
    pause
)
endlocal
