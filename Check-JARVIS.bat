@echo off
REM Reports what is installed and what is missing.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - Checking this machine

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   JARVIS is not installed yet.
    echo   Double-click Install-JARVIS.exe first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m jarvis --preflight
echo.
pause
