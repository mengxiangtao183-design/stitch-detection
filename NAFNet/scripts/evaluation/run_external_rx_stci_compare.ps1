param(
    [string]$OutputDir = "results/stci_external_rx_compare",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

Push-Location $RepoRoot
try {
    python -u scripts/evaluation/evaluate_stitch_clarity.py `
        --data-root H:/DATASET `
        --manifest-dir $ManifestDir `
        --results-root "$RepoRoot/results" `
        --output-dir $OutputDir `
        --domains DB XJ SD RX `
        --max-images $MaxImages
    if ($LASTEXITCODE -ne 0) {
        throw "STCI comparison failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
