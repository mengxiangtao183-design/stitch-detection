param(
    [string]$SynBlurManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_v1_3x_splitval",
    [string]$Img2TestManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_img2test",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_train_img2dev",
    [string]$InputFolder = "Imgs2"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

Push-Location $RepoRoot
try {
    python -u scripts/data_preparation/create_synblur_img2dev_manifests.py `
        --synblur-manifest-dir $SynBlurManifestDir `
        --img2test-manifest-dir $Img2TestManifestDir `
        --manifest-dir $ManifestDir `
        --input-folder $InputFolder
    if ($LASTEXITCODE -ne 0) {
        throw "SynBlur-train Imgs2-dev manifest generation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
