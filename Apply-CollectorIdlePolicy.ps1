$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
. "$PSScriptRoot\Collector-IdleSettings.ps1"
$backup = Join-Path $PSScriptRoot ('data\scheduler-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $backup -Force | Out-Null
foreach ($name in @('MarketData-BTCMinuteSync', 'MarketData-CatalogMinuteSync', 'MarketData-DailySync', 'MarketData-IntradaySync')) {
    $task = Get-ScheduledTask -TaskName $name
    Export-ScheduledTask -TaskName $name | Set-Content -LiteralPath (Join-Path $backup ($name + '.xml')) -Encoding UTF8
    Stop-ScheduledTask -TaskName $name
    $settings = New-CollectorIdleSettings -BudgetMinutes 15
    $settings.ExecutionTimeLimit = $task.Settings.ExecutionTimeLimit
    $settings.Enabled = $task.Settings.Enabled
    $actions = @($task.Actions | ForEach-Object {
        if ($_.Arguments -like '*idle_collector.py*') { $_ }
        else {
            New-ScheduledTaskAction -Execute $collectorPython -Argument ('"' + "$PSScriptRoot\idle_collector.py" + '" "' + $_.Execute + '" ' + $_.Arguments) -WorkingDirectory $PSScriptRoot
        }
    })
    $triggers = $task.Triggers
    if ($name -in @('MarketData-BTCMinuteSync', 'MarketData-CatalogMinuteSync')) {
        $triggers = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
    }
    $description = 'Market data collection requires more than 15 minutes without keyboard/mouse input; stops on activity and restarts when idle.'
    if ($name -in @('MarketData-BTCMinuteSync', 'MarketData-CatalogMinuteSync')) {
        $description += ' Checks every 15 minutes; preserves one-minute candle resolution.'
    }
    $task.Actions = $actions
    $task.Triggers = @($triggers)
    $task.Settings = $settings
    $task.Description = $description
    Set-ScheduledTask -InputObject $task | Out-Null
}
Write-Output "Idle policy applied. Original task definitions: $backup"
