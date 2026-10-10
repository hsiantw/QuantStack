param([string]$DeploymentWorktree = (Join-Path (Split-Path $PSScriptRoot -Parent) 'market-data-render-deploy'))
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
# Complete the daily universe with modest concurrency and low process priority.
# Publish successful updates even when a provider cannot return some tickers.
$reportPath = Join-Path $PSScriptRoot 'data\all-refresh.json'
$previousReport = if (Test-Path -LiteralPath $reportPath) { Get-Content -LiteralPath $reportPath -Raw } else { '' }
& $collectorPython "$PSScriptRoot\refresh_all.py" --daily-only --workers 2
$collectionResult = $LASTEXITCODE
if ($collectionResult -notin @(0, 1)) { exit $collectionResult }
$currentReport = if (Test-Path -LiteralPath $reportPath) { Get-Content -LiteralPath $reportPath -Raw } else { '' }
if (-not $currentReport -or $currentReport -eq $previousReport) { exit 1 }
& $collectorPython "$PSScriptRoot\publish_site.py" --deployment-worktree $DeploymentWorktree
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$collection = $currentReport | ConvertFrom-Json
if (@($collection.failed.PSObject.Properties).Count -gt 0) { exit 1 }
exit 0
