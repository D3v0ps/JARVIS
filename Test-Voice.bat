@echo off
REM Speaks one line, so you know the voice works.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - Testing the voice

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   JARVIS is not installed yet.
    echo   Double-click Install-JARVIS.exe first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m jarvis --say "Good evening, sir. All systems online."
echo.
pause
