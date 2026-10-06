param([ValidateSet('high','low')][string]$Mode = 'low', [switch]$Push)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -Priority 7 -ExecutionTimeLimit (New-TimeSpan -Minutes 55)
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\refresh_hourly.py" + '" --profile ' + $Mode) -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(1) -RepetitionInterval (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName 'MarketData-IntradaySync' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Hourly chart updates ($Mode resource mode)." -Force
# Resolve 09:30 Taipei into the machine's local timezone for Task Scheduler.
$taipei = [TimeZoneInfo]::FindSystemTimeZoneById('Taipei Standard Time')
$taipeiNow = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipei)
$at = [TimeZoneInfo]::ConvertTime([DateTime]::SpecifyKind($taipeiNow.Date.AddHours(9).AddMinutes(30), 'Unspecified'), $taipei, [TimeZoneInfo]::Local)
$commitArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + "$PSScriptRoot\Auto-Commit.ps1" + '"'
if ($Push) { $commitArguments += ' -Push' }
$commitAction = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $commitArguments -WorkingDirectory $PSScriptRoot
$commitSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$commitDescription = 'Daily Git checkpoint of tracked source and chart snapshots at 09:30 Asia/Taipei.'
if ($Push) { $commitDescription += ' Push to origin after committing.' }
Register-ScheduledTask -TaskName 'QuantStack-AutoCommit' -Action $commitAction -Trigger (New-ScheduledTaskTrigger -Daily -At $at) -Principal $principal -Settings $commitSettings -Description $commitDescription -Force
$desktop = [Environment]::GetFolderPath('Desktop')
$launcher = '@echo off' + "`r`n" + 'call "' + "$PSScriptRoot\Update-Charts.bat" + '" %*' + "`r`n"
Set-Content -LiteralPath (Join-Path $desktop 'Update QuantStack Charts.bat') -Value $launcher -Encoding Default
