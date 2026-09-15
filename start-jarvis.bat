@echo off
REM ===========================================================================
REM  J.A.R.V.I.S. - launcher
REM  Creates the virtual environment on first run, installs dependencies,
REM  makes sure Ollama is up, then starts the assistant.
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title J.A.R.V.I.S.

set "PYTHON=python"
set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"

echo.
echo   J.A.R.V.I.S. - starting up
echo   ---------------------------------------------------------------

REM --- 1. Python ------------------------------------------------------------
where %PYTHON% >nul 2>&1
if errorlevel 1 (
    echo   [x] Python was not found on PATH. Install Python 3.13 and try again.
    pause
    exit /b 1
)

REM --- 2. Virtual environment ----------------------------------------------
if not exist "%VPY%" (
    echo   [*] Creating the virtual environment ^(first run only^)...
    %PYTHON% -m venv "%VENV%"
    if errorlevel 1 (
        echo   [x] Could not create the virtual environment.
        pause
        exit /b 1
    )
    echo   [*] Installing dependencies - this takes a few minutes the first time...
    "%VPY%" -m pip install --upgrade pip --quiet
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo   [x] Dependency installation failed. See the output above.
        pause
        exit /b 1
    )
    echo   [*] Running preflight checks...
    "%VPY%" -m jarvis --preflight --fix
)

REM --- 3. Ollama ------------------------------------------------------------
where ollama >nul 2>&1
if errorlevel 1 (
    echo   [!] Ollama is not installed. Install it with:  winget install Ollama.Ollama
    echo       JARVIS will start, but he will have nothing to think with.
) else (
    curl -s -o nul -m 2 http://127.0.0.1:11434/api/tags
    if errorlevel 1 (
        echo   [*] Starting the Ollama service...
        start "" /min ollama serve
        timeout /t 3 /nobreak >nul
    )
)

REM --- 4. Go ----------------------------------------------------------------
echo   [*] Bringing JARVIS online...
echo.
"%VPY%" -m jarvis %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo   JARVIS exited with code %EXITCODE%. The log is in logs\jarvis.log
    pause
)
endlocal & exit /b %EXITCODE%
