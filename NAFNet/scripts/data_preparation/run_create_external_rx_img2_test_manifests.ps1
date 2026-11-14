param(
    [string]$DataRoot = "H:/DATASET",
    [string]$BaseManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_img2test",
    [string]$InputFolder = "Imgs2"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

Push-Location $RepoRoot
try {
    python -u scripts/data_preparation/create_external_rx_img2_test_manifests.py `
        --data-root $DataRoot `
        --base-manifest-dir $BaseManifestDir `
        --manifest-dir $ManifestDir `
        --input-folder $InputFolder
    if ($LASTEXITCODE -ne 0) {
        throw "External-RX Imgs2-test manifest generation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
