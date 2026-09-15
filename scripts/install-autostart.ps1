<#
.SYNOPSIS
    Registers a Task Scheduler task so JARVIS comes online when you log in.

.DESCRIPTION
    Creates a task named "JARVIS" that runs start-jarvis.bat at logon for the
    current user, minimised, with a 15 second delay so the audio devices and the
    Ollama service are ready first. Run this from an ordinary PowerShell window
    in the repository root - no administrator rights required.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-autostart.ps1
    powershell -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1
#>

$ErrorActionPreference = "Stop"

$taskName = "JARVIS"
$root     = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $root "start-jarvis.bat"

if (-not (Test-Path $launcher)) {
    Write-Host "Could not find $launcher - run this from inside the JARVIS repository." -ForegroundColor Red
    exit 1
}

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing the existing '$taskName' task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$action    = New-ScheduledTaskAction -Execute "cmd.exe" `
                                     -Argument "/c start `"`" /min `"$launcher`"" `
                                     -WorkingDirectory $root
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT15S"
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
                                          -DontStopIfGoingOnBatteries `
                                          -StartWhenAvailable `
                                          -ExecutionTimeLimit ([TimeSpan]::Zero)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask -TaskName $taskName `
                       -Description "Brings J.A.R.V.I.S. online at logon." `
                       -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null

Write-Host "JARVIS will now start when you log in." -ForegroundColor Green
Write-Host "Remove it again with: powershell -ExecutionPolicy Bypass -File scripts\uninstall-autostart.ps1"
