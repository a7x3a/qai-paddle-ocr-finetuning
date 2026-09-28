@echo off
setlocal
cd /d "%~dp0"
echo ==============================================================================
echo  Starting Automated Kurdish PaddleOCR Environment Setup
echo ==============================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup encountered an issue. See details above.
    exit /b %ERRORLEVEL%
)
endlocal
