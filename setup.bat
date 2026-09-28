@echo off
setlocal
cd /d "%~dp0"
echo ==============================================================================
echo  Starting Automated Kurdish PaddleOCR Environment Setup
echo ==============================================================================

where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell is required to run automated setup but was not found in PATH.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup encountered an issue. See details above.
    echo %CMDCMDLINE% | find /i "%~0" >nul && pause
    exit /b %ERRORLEVEL%
)

echo %CMDCMDLINE% | find /i "%~0" >nul && pause
endlocal
