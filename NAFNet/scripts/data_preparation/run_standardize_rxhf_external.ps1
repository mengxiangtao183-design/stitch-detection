param([switch]$Overwrite)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Arguments = @(
    (Join-Path $ProjectRoot 'scripts\data_preparation\standardize_rxhf_external.py'),
    '--data-root', 'H:/DATASET',
    '--count', '50', '--size', '320'
)
if ($Overwrite) { $Arguments += '--overwrite' }
& python @Arguments
if ($LASTEXITCODE -ne 0) { throw "RXHF standardization failed with exit code $LASTEXITCODE" }
