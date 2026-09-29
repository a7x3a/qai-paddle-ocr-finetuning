@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo  Kurdish Full-Page Document / PDF OCR Reader
echo ==============================================================================

if "%~1"=="" (
    echo Usage:
    echo   .\read.bat -i ^<image_or_pdf^> [-o ^<output_dir^>] [--dpi 200]
    echo.
    echo Examples:
    echo   .\read.bat -i sample.pdf
    echo   .\read.bat -i document.png -o extracted_pages
    echo.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py read %*
) else (
    python main.py read %*
)

if errorlevel 1 (
    echo [ERROR] Document reader encountered an issue.
    pause
)
endlocal
