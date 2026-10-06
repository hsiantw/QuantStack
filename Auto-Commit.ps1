$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path "$PSScriptRoot\data" -Force | Out-Null
$lock = $null
try {
    $lock = [System.IO.File]::Open("$PSScriptRoot\data\auto-commit.lock", 'OpenOrCreate', 'ReadWrite', 'None')
    function Invoke-Git {
        param([string[]]$GitArgs)
        $output = & git @GitArgs 2>&1
        if ($LASTEXITCODE -ne 0) { throw "git $GitArgs failed: $output" }
        return $output
    }
    # Leave a user's staged work and in-progress Git operations alone.
    foreach ($marker in @('MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply', 'index.lock')) {
        $path = Invoke-Git @('rev-parse', '--git-path', $marker)
        if (Test-Path -LiteralPath $path) { throw "Git operation in progress: $marker" }
    }
    $staged = Invoke-Git @('diff', '--cached', '--name-only')
    if ($staged) { throw 'Auto-commit skipped: the Git index already contains staged changes.' }
    . "$PSScriptRoot\Resolve-CollectorPython.ps1"
    & $collectorPython "$PSScriptRoot\snapshot_charts.py"
    if ($LASTEXITCODE -ne 0) { throw 'Chart snapshot export failed; no commit created.' }
    Invoke-Git @('add', '--', 'chart-snapshots/*.csv', 'chart-snapshots/manifest.json')
    $changes = Invoke-Git @('diff', '--name-only')
    $snapshots = Invoke-Git @('diff', '--cached', '--name-only')
    if ($changes -or $snapshots) {
        Invoke-Git @('add', '-u')
        Invoke-Git @('commit', '-m', ('Scheduled checkpoint ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')))
    } else {
        'No tracked changes to commit.'
    }
} catch {
    Add-Content -LiteralPath "$PSScriptRoot\data\auto-commit.log" -Value ("$(Get-Date -Format o) ERROR: $_")
    throw
} finally {
    if ($lock) { $lock.Dispose() }
}
