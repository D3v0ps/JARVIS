<#
.SYNOPSIS
    Removes the JARVIS logon task created by install-autostart.ps1.
#>

$ErrorActionPreference = "Stop"
$taskName = "JARVIS"

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "JARVIS will no longer start automatically." -ForegroundColor Green
} else {
    Write-Host "No '$taskName' task was registered - nothing to do." -ForegroundColor Yellow
}
