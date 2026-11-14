param(
    [string]$DataRoot = "H:/DATASET",
    [string]$BaseManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_v1_3x_splitval",
    [string]$OutputSubdir = "ImgsSynV1_3x",
    [int]$Seed = 42,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

Push-Location $RepoRoot
try {
    $argsList = @(
        "scripts/data_preparation/create_synblur_v1_3x.py",
        "--data-root", $DataRoot,
        "--base-manifest-dir", $BaseManifestDir,
        "--manifest-dir", $ManifestDir,
        "--output-subdir", $OutputSubdir,
        "--seed", "$Seed"
    )
    if ($Overwrite) {
        $argsList += "--overwrite"
    }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "SynBlur-v1 generation failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
