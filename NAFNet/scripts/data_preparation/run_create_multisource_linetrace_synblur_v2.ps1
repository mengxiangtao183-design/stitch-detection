param(
    [int]$Seed = 42,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Arguments = @(
    (Join-Path $ProjectRoot 'scripts\data_preparation\create_multisource_linetrace_synblur_v2.py'),
    '--data-root', 'H:/DATASET',
    '--base-manifest-dir', 'H:/C3Netqing/MISCFilter/manifests/c3net_external_rx',
    '--manifest-dir', 'H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2',
    '--output-subdir', 'ImgsSynLineV2MS',
    '--seed', $Seed
)
if ($Overwrite) { $Arguments += '--overwrite' }

& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Multi-source LineTrace SynBlur-v2 generation failed with exit code $LASTEXITCODE"
}
