param(
    [string]$CheckpointDir = "checkpoints/MultiSourceV2_MBDNetPPlus_FTPSNR10_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2",
    [string]$OutputDir = "checkpoints/MultiSourceV2_MBDNetPPlus_FTPSNR10_S42_weightavg"
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
$epoch5 = Join-Path $CheckpointDir "model_epoch_5.pth"
$epoch10 = Join-Path $CheckpointDir "model_epoch_10.pth"
foreach ($required in @($epoch5, $epoch10, (Join-Path $ManifestDir "val.tsv"))) {
    if (!(Test-Path -LiteralPath $required)) { throw "Missing required file: $required" }
}

Push-Location $RepoRoot
try {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $average = Join-Path $OutputDir "model_epoch5_epoch10_avg.pth"
    python -u average_c3net_checkpoints.py --checkpoints $epoch5 $epoch10 --output $average
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint averaging failed with exit code $LASTEXITCODE" }
    python -u validate_nafnet_checkpoint_c3.py `
        --data-root H:/DATASET --manifest-dir $ManifestDir --checkpoint $average `
        --arch mbdnet-p-plus --width 64 --workers 0 --output (Join-Path $OutputDir "val_epoch5_epoch10_avg.json")
    if ($LASTEXITCODE -ne 0) { throw "Validation-only evaluation failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
