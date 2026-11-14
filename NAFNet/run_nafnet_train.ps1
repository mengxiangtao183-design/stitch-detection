param(
    [switch]$Smoke,
    [int]$Epochs = 30,
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [int]$CheckpointEvery = 5,
    [int]$LogEverySteps = 25,
    [int]$BatchSize = 4,
    [int]$PatchSize = 320
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

$pythonArgs = @(
    "train_nafnet_c3.py",
    "--data-root", "H:/DATASET",
    "--manifest-dir", "H:/C3Netqing/MISCFilter/manifests/c3net_all",
    "--domains", "DB", "XJ", "SD", "RX",
    "--output-dir", "checkpoints/C3NetAll_NAFNet_width32",
    "--patch-size", "$PatchSize",
    "--batch-size", "$BatchSize",
    "--workers", "0",
    "--epochs", "$Epochs",
    "--checkpoint-every", "$CheckpointEvery",
    "--log-every-steps", "$LogEverySteps"
)

if ($Smoke) {
    $pythonArgs += @("--max-steps", "2", "--max-val-batches", "4")
}

if ($Resume) {
    $pythonArgs += @("--resume", $Resume)
} elseif (!$NoAutoResume) {
    $pythonArgs += "--auto-resume"
}

python -u @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "train_nafnet_c3.py failed with exit code $LASTEXITCODE"
}
