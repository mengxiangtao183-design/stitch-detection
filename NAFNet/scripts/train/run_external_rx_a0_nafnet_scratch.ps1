param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/ExternalRX_A0_NAFNet_Scratch30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx",
    [string]$Resume = "",
    [string]$InitCheckpoint = "",
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}
if (!(Test-Path -LiteralPath (Join-Path $ManifestDir "train.tsv"))) {
    throw "Missing External-RX manifests under $ManifestDir. Run scripts\data_preparation\create_external_rx_manifests.py first."
}

Push-Location $RepoRoot
try {
    $argsList = @(
        "train_nafnet_c3.py",
        "--data-root", "H:/DATASET",
        "--manifest-dir", $ManifestDir,
        "--domains", "DB", "XJ", "SD", "RX",
        "--output-dir", $OutputDir,
        "--patch-size", "320",
        "--batch-size", "$BatchSize",
        "--epochs", "$Epochs",
        "--lr", "0.001",
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
    } elseif ($InitCheckpoint) {
        $argsList += @("--init-checkpoint", $InitCheckpoint)
    } elseif (!$NoAutoResume) {
        $argsList += "--auto-resume"
    }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "External-RX A0 NAFNet training failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
