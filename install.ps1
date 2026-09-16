<#
.SYNOPSIS
    J.A.R.V.I.S. - one-shot installer. Everything, from nothing, in one run.

.DESCRIPTION
    Installs Python and Ollama if they are missing, picks the qwen3 model that fits
    this machine's VRAM, pulls it, builds the virtual environment, installs every
    dependency, downloads the voice and wake-word models, writes the detected values
    into config.yaml, creates Desktop and Start Menu shortcuts, and offers to start
    JARVIS at logon.

    Safe to run twice: every step checks whether it is already done.

.PARAMETER Silent
    No questions. Accepts the defaults (no autostart, launch at the end).

.PARAMETER Swedish
    Also download the Piper Swedish voice.

.PARAMETER Autostart
    Register the logon task without asking.

.PARAMETER NoLaunch
    Do not start JARVIS when the installation finishes.

.PARAMETER Model
    Force a specific Ollama model instead of the one chosen from the VRAM.

.PARAMETER SkipModels
    Skip the Kokoro/wake-word download (useful when they are already in models\).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1
    powershell -ExecutionPolicy Bypass -File install.ps1 -Silent -Swedish -Autostart
#>

[CmdletBinding()]
param(
    [switch]$Silent,
    [switch]$Swedish,
    [switch]$Autostart,
    [switch]$NoLaunch,
    [string]$Model = "",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest and winget far faster

# --------------------------------------------------------------------------- #
#  Paths and logging
# --------------------------------------------------------------------------- #

$Root    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$LogDir  = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir "install.log"
$Venv    = Join-Path $Root ".venv"
$VenvPy  = Join-Path $Venv "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$script:StepNumber = 0
$script:Warnings = New-Object System.Collections.Generic.List[string]

# A terminating error anywhere used to close this window instantly, which told the
# user nothing at all. Catch everything, show it, log it, and wait.
trap {
    Write-Host ""
    Write-Host "  -----------------------------------------------------------" -ForegroundColor Red
    Write-Host "  Something went wrong and the installation stopped." -ForegroundColor Red
    Write-Host ""
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Red
    if ($_.InvocationInfo) {
        Write-Host "  line $($_.InvocationInfo.ScriptLineNumber): $($_.InvocationInfo.Line.Trim())" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "  The full log is in $LogFile" -ForegroundColor Yellow
    Write-Host "  Send that file along and it can be fixed." -ForegroundColor Yellow
    Write-Host "  -----------------------------------------------------------" -ForegroundColor Red
    try { Add-Content -Path $LogFile -Value ("UNHANDLED: " + ($_ | Out-String)) -Encoding utf8 } catch {}
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $line = "{0} {1,-5} {2}" -f (Get-Date -Format "HH:mm:ss"), $Level, $Message
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

function Say {
    param([string]$Message, [string]$Color = "Gray")
    Write-Host "   $Message" -ForegroundColor $Color
    Write-Log $Message
}

function Step {
    param([string]$Title)
    $script:StepNumber++
    Write-Host ""
    Write-Host ("  [{0}] {1}" -f $script:StepNumber, $Title) -ForegroundColor Cyan
    Write-Log "=== STEP $($script:StepNumber): $Title ==="
}

function Ok      { param([string]$m) Say "+ $m" "Green" }
function Warn    { param([string]$m) Say "! $m" "Yellow"; $script:Warnings.Add($m) | Out-Null; Write-Log $m "WARN" }
function Fail    { param([string]$m) Say "x $m" "Red"; Write-Log $m "ERROR" }

function Abort {
    param([string]$Message, [string]$Remedy = "")
    Write-Host ""
    Write-Host "  Installation stopped." -ForegroundColor Red
    Write-Host "  $Message" -ForegroundColor Red
    if ($Remedy) { Write-Host "  $Remedy" -ForegroundColor Yellow }
    Write-Host "  The full log is in $LogFile"
    Write-Log "ABORT: $Message" "ERROR"
    if (-not $Silent) { Write-Host ""; Read-Host "  Press Enter to close" | Out-Null }
    exit 1
}

function Ask {
    param([string]$Question, [bool]$Default = $true)
    if ($Silent) { return $Default }
    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $answer = Read-Host "   $Question $suffix"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer -match '^(y|yes|j|ja)$'
}

# --------------------------------------------------------------------------- #
#  Environment helpers
# --------------------------------------------------------------------------- #

function Update-PathFromRegistry {
    <# winget installs do not touch the PATH of a process that is already running. #>
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ";"
}

function Test-Command {
    <# Returns the path to an executable on PATH, or $null.
       -RejectStoreStub additionally rejects the Microsoft Store's zero-byte
       app-execution aliases. Use it for python (the Store stub only opens the
       Store) but never for winget, whose real entry point IS such an alias. #>
    param([string]$Name, [switch]$RejectStoreStub)
    $cmd = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if (-not $cmd) { return $null }
    if ($RejectStoreStub -and $cmd.Source -and $cmd.Source -like "*WindowsApps*") {
        try {
            if ((Get-Item $cmd.Source -Force).Length -lt 1024) {
                Write-Log "ignoring Microsoft Store stub at $($cmd.Source)"
                return $null
            }
        } catch { return $null }
    }
    return $cmd.Source
}

function Invoke-Native {
    <# Run an executable and return its exit code.

       -Passthrough lets the program write straight to this console, which is what
       you want for anything with a progress bar.

       The ErrorActionPreference dance is not decoration. In Windows PowerShell 5.1
       a native command's stderr, captured with 2>&1, arrives as ErrorRecord objects,
       and with $ErrorActionPreference = 'Stop' the first one is a TERMINATING error.
       'ollama pull' writes its progress bar to stderr, so the installer used to die
       the instant the model download began - and the window closed before anyone
       could read why. #>
    param([string]$File, [string[]]$Arguments, [switch]$Quiet, [switch]$Passthrough)
    Write-Log "run: $File $($Arguments -join ' ')"

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($Passthrough) {
            # Out-Host is not optional. Without it a native command's stdout lands in
            # PowerShell's OUTPUT stream, so this function returns pip's entire
            # transcript with the exit code tacked on the end - and `$code -ne 0`
            # against that array is true even when the command succeeded. That is
            # precisely how a clean `pip install` was reported as a failure.
            & $File @Arguments | Out-Host
        } elseif ($Quiet) {
            & $File @Arguments 2>&1 | ForEach-Object { Write-Log "    $_" }
        } else {
            & $File @Arguments 2>&1 | ForEach-Object {
                Write-Host "      $_" -ForegroundColor DarkGray
                Write-Log "    $_"
            }
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($code -is [array]) { $code = $code[-1] }     # belt and braces
    if ($null -eq $code) { $code = 0 }
    $code = [int]$code
    Write-Log "exit code $code"
    return $code
}

function Install-WithWinget {
    param([string]$Id, [string]$Friendly)
    if (-not (Test-Command "winget")) {
        Abort "$Friendly is missing and winget is not available on this machine." `
              "Install '$Friendly' by hand, then run this installer again."
    }
    Say "Installing $Friendly (this can take a few minutes)..."
    $wingetArgs = @("install", "--id", $Id, "--exact", "--silent",
                    "--accept-package-agreements", "--accept-source-agreements",
                    "--disable-interactivity")
    $code = Invoke-Native "winget" $wingetArgs -Quiet
    # 0 = installed, -1978335189 = already installed, 0x8A15002B = no applicable upgrade
    if ($code -ne 0 -and $code -ne -1978335189 -and $code -ne -1978335212) {
        Write-Log "winget exit code $code for $Id" "WARN"
    }
    Update-PathFromRegistry
}

# --------------------------------------------------------------------------- #
#  Model selection
# --------------------------------------------------------------------------- #

function Select-Model {
    <# Pick the qwen3 model and the Whisper size that fit this machine.

       The language model and Whisper share the VRAM, so the model can only be as
       big as what is left after speech recognition. These thresholds leave a
       comfortable margin: qwen3:14b is about 9 GB and Whisper medium about 3 GB,
       which is why 14b needs a 16 GB card rather than a 12 GB one.

       Returns a hashtable: Model, Whisper, Reason. #>
    param([double]$VramGb, [string]$Override = "")

    if ($Override) {
        $whisper = if ($VramGb -ge 10) { "medium" } elseif ($VramGb -ge 6) { "small" } else { "base" }
        return @{ Model = $Override; Whisper = $whisper; Embed = "qwen3-embedding:0.6b"
                  Reason = "Using the model you asked for: $Override" }
    }
    if ($VramGb -ge 15) {
        # Deliberately not qwen3:14b. 14b is about 8.3 GB and Whisper medium another
        # 2-3, which leaves no room for the embedding model that semantic memory and
        # document search need. For a voice assistant, 8b plus a memory that grows is
        # a better machine than 14b with none: 14b is sharper on hard questions,
        # slower on the ninety percent that are not.
        return @{ Model = "qwen3:8b"; Whisper = "medium"; Embed = "qwen3-embedding:0.6b"
                  Reason = "$VramGb GB of VRAM - taking qwen3:8b with Whisper medium, leaving room for the embedding model that gives him a memory." }
    }
    if ($VramGb -ge 7) {
        return @{ Model = "qwen3:8b"; Whisper = "small"; Embed = "qwen3-embedding:0.6b"
                  Reason = "$VramGb GB of VRAM - taking qwen3:8b with Whisper small, which keeps the whole turn under two seconds." }
    }
    if ($VramGb -ge 4) {
        return @{ Model = "qwen3:4b"; Whisper = "base"; Embed = ""
                  Reason = "$VramGb GB of VRAM - taking the smaller qwen3:4b so it fits on the card." }
    }
    return @{ Model = "qwen3:4b"; Whisper = "base"; Embed = ""
              Reason = "No usable GPU - taking qwen3:4b on the CPU. He will be slower, but he will work." }
}

# --------------------------------------------------------------------------- #
#  Banner
# --------------------------------------------------------------------------- #

Clear-Host
Write-Host ""
Write-Host "   +--------------------------------------------------------------+" -ForegroundColor DarkCyan
Write-Host "   |                                                              |" -ForegroundColor DarkCyan
Write-Host "   |      J . A . R . V . I . S .   -   i n s t a l l e r         |" -ForegroundColor Cyan
Write-Host "   |                                                              |" -ForegroundColor DarkCyan
Write-Host "   |      Just A Rather Very Intelligent System                   |" -ForegroundColor DarkCyan
Write-Host "   |      Everything runs on this machine. Nothing phones home.   |" -ForegroundColor DarkCyan
Write-Host "   |                                                              |" -ForegroundColor DarkCyan
Write-Host "   +--------------------------------------------------------------+" -ForegroundColor DarkCyan
Write-Host ""

Write-Log "---------------------------------------------------------------"
Write-Log "JARVIS installer started in $Root"
Write-Log "PowerShell $($PSVersionTable.PSVersion) on $([Environment]::OSVersion.VersionString)"

if (-not (Test-Path (Join-Path $Root "requirements.txt"))) {
    Abort "This script is not in the JARVIS folder (requirements.txt is missing)." `
          "Unpack the whole repository and run the installer from inside it."
}

$isAdmin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
            [Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Log "Administrator: $isAdmin"

# --------------------------------------------------------------------------- #
#  1. Windows
# --------------------------------------------------------------------------- #

Step "Checking Windows"
$os = Get-CimInstance Win32_OperatingSystem
Say "$($os.Caption) (build $($os.BuildNumber))"
if ([int]$os.BuildNumber -lt 19041) {
    Warn "This was built for Windows 11. Older builds may misbehave."
}
$ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Say "$ramGb GB of RAM"
if ($ramGb -lt 8) { Warn "8 GB of RAM or more is strongly recommended." }

$freeGb = 999.0
try {
    if ($Root -match '^[A-Za-z]:') {
        $freeGb = [math]::Round((Get-PSDrive -Name $Root.Substring(0,1)).Free / 1GB, 1)
        Say "$freeGb GB free on this drive"
    } else {
        Say "Installing to a network path - skipping the free space check"
    }
} catch { Write-Log "free space check failed: $_" "WARN" }
if ($freeGb -lt 15) {
    Warn "The model, the voice and the dependencies need roughly 15 GB. You have $freeGb GB."
    if (-not (Ask "Continue anyway?" $false)) { Abort "Not enough free disk space." }
}

# --------------------------------------------------------------------------- #
#  2. Python
# --------------------------------------------------------------------------- #

Step "Python"
$pythonExe = Test-Command "python" -RejectStoreStub
$pythonOk = $false
if ($pythonExe) {
    try {
        $verText = (& $pythonExe --version 2>&1) -join " "
        if ($verText -match '(\d+)\.(\d+)\.(\d+)') {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 10) {
                Ok "$verText at $pythonExe"
                $pythonOk = $true
            } else {
                Warn "$verText is too old - JARVIS needs Python 3.10 or newer."
            }
        }
    } catch { Warn "Could not run the Python on PATH: $_" }
}

if (-not $pythonOk) {
    Say "Python 3.13 is not installed. Fetching it..."
    Install-WithWinget "Python.Python.3.13" "Python 3.13"
    $pythonExe = Test-Command "python" -RejectStoreStub
    if (-not $pythonExe) {
        # winget sometimes only puts it in the user launcher directory
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
        if (Test-Path $candidate) { $pythonExe = $candidate }
    }
    if (-not $pythonExe) {
        Abort "Python was installed but is still not on PATH." `
              "Close this window, open a new one, and run the installer again."
    }
    Ok "$((& $pythonExe --version 2>&1) -join ' ')"
}

# --------------------------------------------------------------------------- #
#  3. Graphics card -> which model fits
# --------------------------------------------------------------------------- #

Step "Graphics card"
$vramGb = 0.0
$gpuName = ""
if (Test-Command "nvidia-smi") {
    try {
        $line = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>&1 | Select-Object -First 1)
        if ($line -match '^(.*),\s*(\d+)\s*$') {
            $gpuName = $Matches[1].Trim()
            $vramGb = [math]::Round([double]$Matches[2] / 1024, 1)
            Ok "$gpuName with $vramGb GB of VRAM"
        }
    } catch { Warn "nvidia-smi is present but did not answer: $_" }
}
if ($vramGb -eq 0) {
    Warn "No NVIDIA GPU detected - JARVIS will run on the CPU. He will be slower but he will work."
}

$choice = Select-Model -VramGb $vramGb -Override $Model
$chosenModel  = $choice.Model
$whisperModel = $choice.Whisper
$embedModel   = $choice.Embed
Say $choice.Reason

# --------------------------------------------------------------------------- #
#  4. Ollama
# --------------------------------------------------------------------------- #

Step "Ollama"
if (Test-Command "ollama") {
    # `ollama --version` also prints a warning when the service is not up yet, and
    # that warning arrives on stderr - so capture it with the preference relaxed and
    # keep only the line that actually carries the version.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $versionLines = @(& ollama --version 2>&1 | ForEach-Object { "$_" }) }
    finally { $ErrorActionPreference = $previousPreference }
    $version = $versionLines | Where-Object { $_ -match "version" } | Select-Object -First 1
    if (-not $version) { $version = "installed" }
    Ok "Ollama is already installed ($($version.Trim()))"
} else {
    Install-WithWinget "Ollama.Ollama" "Ollama"
    if (-not (Test-Command "ollama")) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $candidate) {
            $env:Path = "$env:Path;$(Split-Path $candidate)"
        }
    }
    if (-not (Test-Command "ollama")) {
        Abort "Ollama was installed but is not on PATH yet." `
              "Close this window, open a new one, and run the installer again."
    }
    Ok "Ollama installed"
}

Say "Making sure the Ollama service is answering..."
$daemonUp = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 -UseBasicParsing | Out-Null
        $daemonUp = $true; break
    } catch {
        if ($i -eq 0) {
            Say "Starting it..."
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        }
        Start-Sleep -Seconds 1
    }
}
if ($daemonUp) { Ok "Ollama is running on port 11434" }
else { Abort "The Ollama service never answered on http://127.0.0.1:11434." "Try running 'ollama serve' in a terminal and watch what it says." }

# --------------------------------------------------------------------------- #
#  5. The model
# --------------------------------------------------------------------------- #

Step "Language model ($chosenModel)"
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try { $installed = (& ollama list 2>&1 | Out-String) } finally { $ErrorActionPreference = $previousPreference }
if ($installed -match [regex]::Escape($chosenModel)) {
    Ok "$chosenModel is already pulled"
} else {
    Say "Pulling $chosenModel - this is the big one, several gigabytes. Time for a coffee."
    $code = Invoke-Native "ollama" @("pull", $chosenModel) -Passthrough
    if ($code -ne 0) {
        Warn "Pulling $chosenModel failed. Falling back to qwen3:8b."
        $chosenModel = "qwen3:8b"
        $code = Invoke-Native "ollama" @("pull", $chosenModel) -Passthrough
        if ($code -ne 0) { Abort "Could not pull a language model." "Check your internet connection and run 'ollama pull qwen3:8b' by hand." }
    }
    Ok "$chosenModel is ready"
}

if ($embedModel) {
    if ($installed -match [regex]::Escape($embedModel)) {
        Ok "$embedModel is already pulled"
    } else {
        Say "Pulling $embedModel - the memory model, a few hundred megabytes."
        $code = Invoke-Native "ollama" @("pull", $embedModel) -Passthrough
        if ($code -ne 0) { Warn "Could not pull $embedModel; semantic memory will be unavailable." }
        else { Ok "$embedModel is ready" }
    }
}

# --------------------------------------------------------------------------- #
#  6. Virtual environment and dependencies
# --------------------------------------------------------------------------- #

Step "Python environment"
if (Test-Path $VenvPy) {
    Ok "The virtual environment already exists"
} else {
    Say "Creating .venv ..."
    $code = Invoke-Native $pythonExe @("-m", "venv", $Venv) -Quiet
    if ($code -ne 0 -or -not (Test-Path $VenvPy)) { Abort "Could not create the virtual environment." }
    Ok "Virtual environment created"
}

Say "Installing dependencies - the first run takes a few minutes..."
Invoke-Native $VenvPy @("-m", "pip", "install", "--upgrade", "pip", "--quiet") -Quiet | Out-Null
$code = Invoke-Native $VenvPy @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt")) -Passthrough
if ($code -ne 0) {
    Abort "Dependency installation failed." "The last lines of $LogFile say why."
}
Ok "All dependencies installed"

# --------------------------------------------------------------------------- #
#  7. Voice and wake word models
# --------------------------------------------------------------------------- #

if (-not $SkipModels) {
    Step "Voice and wake word"
    $fetchArgs = @((Join-Path $Root "scripts\fetch_models.py"))
    if ($Swedish) { $fetchArgs += "--swedish" }
    $code = Invoke-Native $VenvPy $fetchArgs -Passthrough
    if ($code -ne 0) { Warn "Some model files did not download. JARVIS will fall back to the Windows voice until you run scripts\fetch_models.py again." }
    else { Ok "Kokoro voice and wake word models are in place" }
}

# --------------------------------------------------------------------------- #
#  8. Write what we found into config.yaml
# --------------------------------------------------------------------------- #

Step "Configuration"
$configArgs = @("-m", "jarvis", "--preflight", "--fix", "--set-model", $chosenModel, "--set-whisper", $whisperModel)
$code = Invoke-Native $VenvPy $configArgs
if ($code -ne 0) { Warn "The preflight check reported problems - see above. JARVIS will still start." }
else { Ok "config.yaml now matches this machine" }

# --------------------------------------------------------------------------- #
#  9. Shortcuts
# --------------------------------------------------------------------------- #

Step "Shortcuts"
$launcherExe = Join-Path $Root "JARVIS.exe"
$launcherBat = Join-Path $Root "start-jarvis.bat"
$launcher = if (Test-Path $launcherExe) { $launcherExe } else { $launcherBat }
$iconPath = Join-Path $Root "assets\jarvis.ico"
if (-not (Test-Path $iconPath)) { $iconPath = "$env:SystemRoot\System32\shell32.dll,13" }

function New-Shortcut {
    param([string]$Path, [string]$Description)
    try {
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($Path)
        $sc.TargetPath = $launcher
        $sc.WorkingDirectory = $Root
        $sc.Description = $Description
        $sc.IconLocation = $iconPath
        $sc.WindowStyle = 7          # minimised - the overlay is the real interface
        $sc.Save()
        return $true
    } catch {
        Write-Log "shortcut failed: $_" "WARN"
        return $false
    }
}

$desktop = [Environment]::GetFolderPath("Desktop")
if (New-Shortcut (Join-Path $desktop "JARVIS.lnk") "Bring J.A.R.V.I.S. online") { Ok "Desktop shortcut created" }
else { Warn "Could not create the Desktop shortcut" }

$startMenu = Join-Path ([Environment]::GetFolderPath("ApplicationData")) "Microsoft\Windows\Start Menu\Programs"
if (New-Shortcut (Join-Path $startMenu "JARVIS.lnk") "Bring J.A.R.V.I.S. online") { Ok "Start Menu shortcut created" }
else { Warn "Could not create the Start Menu shortcut" }

# --------------------------------------------------------------------------- #
#  10. Autostart
# --------------------------------------------------------------------------- #

Step "Start at logon"
$wantAutostart = $Autostart
if (-not $Autostart -and -not $Silent) {
    $wantAutostart = Ask "Should JARVIS come online automatically when you log in?" $false
}
if ($wantAutostart) {
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\install-autostart.ps1") | ForEach-Object { Write-Log "    $_" }
        Ok "JARVIS will start when you log in"
    } catch { Warn "Could not register the logon task: $_" }
} else {
    Say "Skipped - you can do it later with scripts\install-autostart.ps1"
}

# --------------------------------------------------------------------------- #
#  Done
# --------------------------------------------------------------------------- #

Write-Host ""
Write-Host "   +--------------------------------------------------------------+" -ForegroundColor Green
Write-Host "   |   JARVIS is installed and ready.                             |" -ForegroundColor Green
Write-Host "   +--------------------------------------------------------------+" -ForegroundColor Green
Write-Host ""
Write-Host "   Model      : $chosenModel" -ForegroundColor Gray
Write-Host "   Whisper    : $whisperModel" -ForegroundColor Gray
if ($gpuName) { Write-Host "   GPU        : $gpuName ($vramGb GB)" -ForegroundColor Gray }
Write-Host "   Start it   : double-click JARVIS on your Desktop" -ForegroundColor Gray
Write-Host ""
Write-Host "   Then say:" -ForegroundColor White
Write-Host '      "Hey Jarvis."          ' -NoNewline -ForegroundColor Cyan; Write-Host "(wait for the chime)" -ForegroundColor DarkGray
Write-Host '      "What time is it and how is the system?"' -ForegroundColor Cyan
Write-Host '      "Open Spotify."' -ForegroundColor Cyan
Write-Host ""

if ($script:Warnings.Count -gt 0) {
    Write-Host "   Worth knowing:" -ForegroundColor Yellow
    foreach ($w in $script:Warnings) { Write-Host "     - $w" -ForegroundColor Yellow }
    Write-Host ""
}
Write-Log "Installation finished with $($script:Warnings.Count) warning(s)"

if (-not $NoLaunch) {
    if (Ask "Start JARVIS now?" $true) {
        Say "Bringing him online..."
        Start-Process -FilePath $launcher -WorkingDirectory $Root
        Start-Sleep -Seconds 2
        exit 0
    }
}

if (-not $Silent) { Write-Host ""; Read-Host "   Press Enter to close" | Out-Null }
exit 0
