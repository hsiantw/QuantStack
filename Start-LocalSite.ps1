param(
    [ValidateRange(1024, 65535)][int]$Port = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$address = "http://127.0.0.1:$Port"
$logDirectory = Join-Path $PSScriptRoot 'data'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

function Test-LocalSite {
    try {
        $response = Invoke-WebRequest -Uri "$address/workspace/" -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200 -and $response.Content.Contains('id="chart"') -and $response.Content.Contains('QuantStack')
    } catch { return $false }
}

if (-not (Test-LocalSite)) {
    $candidates = @(
        (Join-Path $PSScriptRoot '.venv-integration\Scripts\python.exe'),
        (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
    )
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython) { $candidates += $systemPython.Source }
    $previewPython = $null
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        # Probe in a child process so a stale virtual environment cannot stop fallback.
        $probe = Start-Process -FilePath $candidate -ArgumentList '-c "import aiohttp"' -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$logDirectory\preview-runtime.log" -RedirectStandardError "$logDirectory\preview-runtime-errors.log"
        if ($probe.ExitCode -eq 0) { $previewPython = $candidate; break }
    }
    if (-not $previewPython) {
        throw 'No working Python environment with aiohttp. Install requirements.txt in a virtual environment, then retry.'
    }
    $arguments = '"' + (Join-Path $PSScriptRoot 'serve_quantstack.py') + '" --host 127.0.0.1 --port ' + $Port
    $process = Start-Process -FilePath $previewPython -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logDirectory\local-site-$Port.log" -RedirectStandardError "$logDirectory\local-site-$Port-errors.log"
    $deadline = (Get-Date).AddSeconds(90)
    while (-not (Test-LocalSite)) {
        $process.Refresh()
        if ($process.HasExited) {
            throw "Local preview exited. Check data\local-site-$Port-errors.log. If port $Port is busy, use -Port 8502."
        }
        if ((Get-Date) -gt $deadline) {
            throw "Preview is still starting. Check data\local-site-$Port.log and retry shortly."
        }
        Start-Sleep -Milliseconds 500
    }
}

Write-Host "Local preview: $address"
Write-Host 'Analysis tools are integrated into the chart workspace right rail.'
Write-Host 'Refresh the browser after frontend edits. Restart the preview after Python changes.'
if (-not $NoBrowser) { Start-Process $address }
