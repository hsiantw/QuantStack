param(
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [switch]$OpenBrowser
)

$ErrorActionPreference = 'Stop'
$taskName = 'QuantStack-LocalWebsite'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$launcher = Join-Path $PSScriptRoot 'Start-LocalSite.ps1'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Local website launcher is missing: $launcher"
}

$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $launcher + '" -Port ' + $Port
if (-not $OpenBrowser) { $arguments += ' -NoBrowser' }
$action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = 'PT15S'
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
# The launcher starts a background server. Do not impose a runtime limit that
# could terminate that server, or stop it when this computer switches to battery.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "Start the local QuantStack website at http://127.0.0.1:$Port after this user signs in. Reuse an already running server." `
    -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host "Installed ${taskName}: starts 15 seconds after $user signs in."
Write-Host "Local website: http://127.0.0.1:$Port/workspace/"
if (-not $OpenBrowser) { Write-Host 'Runs in the background; browser auto-open is disabled.' }
