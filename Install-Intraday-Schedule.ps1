$ErrorActionPreference = 'Stop'
$action = New-ScheduledTaskAction -Execute "$PSScriptRoot\.venv\Scripts\python.exe" -Argument ('"' + "$PSScriptRoot\market_data.py" + '" intraday') -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(1) -RepetitionInterval (New-TimeSpan -Hours 1)
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 59) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'MarketData-IntradaySync' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Refresh native hourly Yahoo Finance bars for the expanded symbol universe every hour.' -Force
