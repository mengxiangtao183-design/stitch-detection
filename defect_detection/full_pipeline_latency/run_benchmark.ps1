param([switch]$Smoke, [switch]$AllPaths)
$ErrorActionPreference = 'Stop'
$latencyPython = 'D:\professional_software\anaconda3\envs\c3net\python.exe'
$latencyScript = Join-Path $PSScriptRoot 'benchmark_full_pipeline.py'
$latencyConfig = Join-Path $PSScriptRoot 'config.json'
if (-not (Test-Path -LiteralPath $latencyPython)) { throw "Python not found: $latencyPython" }
& $latencyPython $latencyScript --config $latencyConfig --check-only
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
$latencyExtra = @()
if ($Smoke) { $latencyExtra = @('--max-images', '2', '--repeats', '1', '--warmup', '2') }
& $latencyPython $latencyScript --config $latencyConfig --policy routed --workload mixed @latencyExtra
if ($LASTEXITCODE -ne 0) { throw 'Routed benchmark failed' }
if ($AllPaths -and -not $Smoke) {
    & $latencyPython $latencyScript --config $latencyConfig --policy always_deblur --workload blurred
    if ($LASTEXITCODE -ne 0) { throw 'Full deblur path benchmark failed' }
    & $latencyPython $latencyScript --config $latencyConfig --policy bypass --workload clear
    if ($LASTEXITCODE -ne 0) { throw 'Bypass path benchmark failed' }
}
