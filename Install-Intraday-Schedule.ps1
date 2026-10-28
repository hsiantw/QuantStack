$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
. "$PSScriptRoot\Collector-IdleSettings.ps1"
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\market_data.py" + '" intraday') -WorkingDirectory $PSScriptRoot
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\idle_collector.py" + '" "' + $action.Execute + '" ' + $action.Arguments) -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(1) -RepetitionInterval (New-TimeSpan -Hours 1)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-CollectorIdleSettings -BudgetMinutes 15
Register-ScheduledTask -TaskName 'MarketData-IntradaySync' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Refresh native hourly Yahoo Finance bars for the expanded symbol universe every hour.' -Force
