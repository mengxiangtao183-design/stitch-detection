param(
    [switch]$Smoke,
    [int]$Epochs = 100,
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [int]$SaveEverySteps = 0,
    [int]$CheckpointEvery = 5,
    [int]$LogEverySteps = 25
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

$pythonArgs = @(
    "train_c3net.py",
    "--data-root", "H:/DATASET",
    "--domains", "DB", "XJ", "SD", "RX",
    "--patch-size", "320",
    "--batch-size", "4",
    "--workers", "0",
    "--epochs", "$Epochs",
    "--save-every-steps", "$SaveEverySteps",
    "--checkpoint-every", "$CheckpointEvery",
    "--log-every-steps", "$LogEverySteps"
)

if ($Smoke) {
    $pythonArgs += @("--max-steps", "10", "--max-val-batches", "4")
}

if ($Resume) {
    $pythonArgs += @("--resume", $Resume)
} elseif (!$NoAutoResume) {
    $pythonArgs += "--auto-resume"
}

python -u @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "train_c3net.py failed with exit code $LASTEXITCODE"
}
