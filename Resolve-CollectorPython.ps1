$collectorCandidates = @(
    (Join-Path $PSScriptRoot '.venv-integration\Scripts\python.exe'),
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)
$collectorSystem = Get-Command python -ErrorAction SilentlyContinue
if ($collectorSystem) { $collectorCandidates += $collectorSystem.Source }
$collectorPython = $null
foreach ($candidate in $collectorCandidates) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    & $candidate -c 'import yfinance, pandas' 2>$null
    if ($LASTEXITCODE -eq 0) { $collectorPython = $candidate; break }
}
if (-not $collectorPython) { throw 'Install requirements.txt in a working Python environment first.' }
