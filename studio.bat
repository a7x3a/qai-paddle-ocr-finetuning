@echo off
setlocal
cd /d "%~dp0"
echo ==============================================================================
echo  Launching Kurdish Full-Page PaddleOCR Web Studio
echo ==============================================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py serve %*
) else (
    python main.py serve %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Web Studio exited with code %ERRORLEVEL%.
    pause
)
endlocal
