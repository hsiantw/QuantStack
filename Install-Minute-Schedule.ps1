$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
. "$PSScriptRoot\Collector-IdleSettings.ps1"
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\refresh_minutes.py" + '"') -WorkingDirectory $PSScriptRoot
$action = New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\idle_collector.py" + '" "' + $action.Execute + '" ' + $action.Arguments) -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-CollectorIdleSettings -BudgetMinutes 3
Register-ScheduledTask -TaskName 'MarketData-CatalogMinuteSync' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Rotate one-minute candle pulls across all local Stocks/Crypto catalog symbols; BTC uses its separate Bitstamp task. Runs after 15 minutes idle on AC power; stops on input.' -Force
Start-ScheduledTask -TaskName 'MarketData-CatalogMinuteSync'
