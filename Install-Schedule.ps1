$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
. "$PSScriptRoot\Collector-IdleSettings.ps1"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + "$PSScriptRoot\Refresh-And-Publish.ps1" + '"') -WorkingDirectory $PSScriptRoot
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\idle_collector.py" + '" "' + $action.Execute + '" ' + $action.Arguments) -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Daily -At '09:00'
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-CollectorIdleSettings -BudgetMinutes 240
Register-ScheduledTask -TaskName 'MarketData-DailySync' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Update priority stock and crypto daily data and publish the snapshot to Render at 09:00 local time.' -Force
