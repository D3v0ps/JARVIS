@echo off
REM Downloads the British Kokoro voice and the wake-word models.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - fetching the voice

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   JARVIS is not installed yet.
    echo   Double-click Install-JARVIS.exe first.
    echo.
    pause
    exit /b 1
)

echo.
echo   Fetching the Kokoro voice ^(about 340 MB^) and the wake word models.
echo   Add the word swedish after the filename if you also want the Swedish voice.
echo.
".venv\Scripts\python.exe" scripts\fetch_models.py %*
echo.
pause
