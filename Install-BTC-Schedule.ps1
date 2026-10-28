$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
. "$PSScriptRoot\Collector-IdleSettings.ps1"
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\btc_backfill.py" + '"') -WorkingDirectory $PSScriptRoot
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\idle_collector.py" + '" "' + $action.Execute + '" ' + $action.Arguments) -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-CollectorIdleSettings -BudgetMinutes 15
Register-ScheduledTask -TaskName 'MarketData-BTCMinuteSync' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Catch up completed BTC/USD one-minute Bitstamp candles every 15 minutes after 15 minutes idle. Resumes checkpoints, runs while logged in on AC power, and prevents overlapping runs.' -Force
Start-ScheduledTask -TaskName 'MarketData-BTCMinuteSync'
