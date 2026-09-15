@echo off
REM Cycles the arc reactor through every state for ten seconds.
REM Double-click this file - no terminal needed.
cd /d "%~dp0"
title J.A.R.V.I.S. - Showing the overlay

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   JARVIS is not installed yet.
    echo   Double-click Install-JARVIS.exe first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m jarvis --overlay-test
echo.
pause
