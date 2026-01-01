@echo off
REM Language Translation Tool - Quick Start Script
REM This script will open the translation tool in your default browser

echo.
echo ========================================
echo  Language Translation Tool - Quick Start
echo ========================================
echo.

REM Get the current directory
set SCRIPT_DIR=%~dp0

echo Opening the translation tool...
echo.

REM Open the index.html file with the default browser
start "" "%SCRIPT_DIR%index.html"

echo.
echo The translation tool should now open in your browser.
echo.
echo To use the tool:
echo 1. Enter text in the "Text to Translate" field
echo 2. Select source and target languages
echo 3. Click "Translate" or press Ctrl+Enter
echo.
echo Features:
echo - Copy: Copy translated text to clipboard
echo - Speak: Listen to the translation (Text-to-Speech)
echo - Swap: Switch between source and target languages
echo.
pause
