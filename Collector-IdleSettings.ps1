# Shared by every market-data task installer. Preserve on reinstallation.
function New-CollectorIdleSettings([int]$BudgetMinutes = 15) {
    New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -Priority 7 `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $BudgetMinutes) -RunOnlyIfIdle `
        -IdleDuration (New-TimeSpan -Minutes 15) -IdleWaitTimeout (New-TimeSpan -Hours 24) `
        -RestartOnIdle
    # StopOnIdleEnd is enabled by default (do not pass -DontStopOnIdleEnd).
}
