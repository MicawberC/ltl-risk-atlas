$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-repro\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Create .venv-repro and install reproduction/requirements-frozen.txt first.'
}

$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Push-Location $projectRoot
try {
    $checks = @(
        @('iros_core.py', 'iros-core.log'),
        @('policy_extraction_probe.py', 'policy-extraction-probe.log'),
        @('policy_extraction_fixed_bounds_probe.py', 'policy-extraction-fixed-bounds.log'),
        @('iros_demo_smoke.py', 'iros-demo.log')
    )
    foreach ($check in $checks) {
        & $python (Join-Path $PSScriptRoot $check[0]) *> (Join-Path $logDir $check[1])
        if ($LASTEXITCODE -ne 0) { throw "$($check[0]) failed; see reproduction/logs/$($check[1])." }
    }
    Write-Output 'IROS public example checks complete. See reproduction/logs/ and reproduction/*_results.json.'
}
finally { Pop-Location }
