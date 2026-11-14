param(
    [int]$Epochs = 15,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/Ablation_A0_NAFNet_FT15_S42",
    [string]$InitCheckpoint = "checkpoints/C3NetAll_NAFNet_width32/model_epoch_30.pth",
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}
if (!(Test-Path -LiteralPath (Join-Path $PSScriptRoot $InitCheckpoint))) {
    throw "Initial checkpoint not found: $InitCheckpoint"
}

$argsList = @(
    "train_nafnet_c3.py",
    "--data-root", "H:/DATASET",
    "--manifest-dir", "H:/C3Netqing/MISCFilter/manifests/c3net_all",
    "--domains", "DB", "XJ", "SD", "RX",
    "--output-dir", $OutputDir,
    "--init-checkpoint", $InitCheckpoint,
    "--patch-size", "320",
    "--batch-size", "$BatchSize",
    "--epochs", "$Epochs",
    "--lr", "0.0002",
    "--min-lr", "0.0000001",
    "--weight-decay", "0.001",
    "--val-every", "5",
    "--workers", "0",
    "--seed", "42",
    "--checkpoint-every", "5",
    "--log-every-steps", "25"
)

if ($Smoke) {
    $argsList += @("--max-steps", "2", "--max-val-batches", "4")
}
if ($Resume) {
    $argsList += @("--resume", $Resume)
} elseif (!$NoAutoResume) {
    $argsList += "--auto-resume"
}

python -u @argsList
if ($LASTEXITCODE -ne 0) {
    throw "A0 NAFNet FT15 training failed with exit code $LASTEXITCODE"
}
