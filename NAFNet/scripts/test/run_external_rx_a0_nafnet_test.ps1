param(
    [string]$Checkpoint = "checkpoints/ExternalRX_A0_NAFNet_Scratch30_S42/model_best.pth",
    [string]$OutputDir = "results/ExternalRX_A0_NAFNet_Scratch30_S42_midroi_img2test",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_img2test",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

Push-Location $RepoRoot
try {
    python -u test_nafnet_c3.py `
        --data-root H:/DATASET `
        --manifest-dir $ManifestDir `
        --domains DB XJ SD RX `
        --checkpoint $Checkpoint `
        --output-dir $OutputDir `
        --window-size 320 `
        --workers 0 `
        --max-images $MaxImages
    if ($LASTEXITCODE -ne 0) {
        throw "External-RX A0 NAFNet test failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
