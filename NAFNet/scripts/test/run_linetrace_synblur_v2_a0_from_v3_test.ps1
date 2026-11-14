param(
    [string]$Checkpoint = "checkpoints/LineTraceSynBlurV2_A0_FromV3FT20_S42/model_best.pth",
    [string]$OutputDir = "results/LineTraceSynBlurV2_A0_FromV3FT20_S42_midroi",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_controlled_linetrace_synblur_v2",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
if (!(Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
Push-Location $RepoRoot
try {
    python -u test_nafnet_c3.py --arch nafnet `
        --data-root H:/DATASET --manifest-dir $ManifestDir --domains DB XJ SD RX `
        --checkpoint $Checkpoint --output-dir $OutputDir --window-size 320 --workers 0 --max-images $MaxImages `
        --width 32 --enc-blks 1 1 1 28 --middle-blks 1 --dec-blks 1 1 1 1
    if ($LASTEXITCODE -ne 0) { throw "LineTrace SynBlur-v2 A0-from-V3 test failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
