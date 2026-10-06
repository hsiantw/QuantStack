param([ValidateSet('high','low')][string]$Mode = 'low')
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
& $collectorPython "$PSScriptRoot\refresh_hourly.py" --profile $Mode
exit $LASTEXITCODE
