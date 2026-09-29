@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py infer %*
) else (
    python main.py infer %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Inference exited with code %ERRORLEVEL%.
    pause
)
endlocal
