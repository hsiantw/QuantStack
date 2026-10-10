param([string]$DeploymentWorktree = (Join-Path (Split-Path $PSScriptRoot -Parent) 'market-data-render-deploy'))
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Resolve-CollectorPython.ps1"
# Preserve the configured low-resource collection budget. Partial successful
# updates must still reach Render instead of remaining only in the local DB.
$reportPath = Join-Path $PSScriptRoot 'data\daily-refresh.json'
$previousReport = if (Test-Path -LiteralPath $reportPath) { Get-Content -LiteralPath $reportPath -Raw } else { '' }
& $collectorPython "$PSScriptRoot\market_data.py" sync --local-only
$collectionResult = $LASTEXITCODE
if ($collectionResult -notin @(0, 1)) { exit $collectionResult }
$currentReport = if (Test-Path -LiteralPath $reportPath) { Get-Content -LiteralPath $reportPath -Raw } else { '' }
if (-not $currentReport -or $currentReport -eq $previousReport) { exit 1 }
& $collectorPython "$PSScriptRoot\publish_site.py" --deployment-worktree $DeploymentWorktree
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$collection = $currentReport | ConvertFrom-Json
if ($collection.failed.Count -gt 0) { exit 1 }
exit 0
