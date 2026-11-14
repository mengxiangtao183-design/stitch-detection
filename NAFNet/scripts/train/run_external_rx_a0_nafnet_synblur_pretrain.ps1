param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/ExternalRX_A0_NAFNet_SynBlurV1SplitValPre30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_v1_3x_splitval",
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$BaseScript = Join-Path $PSScriptRoot "run_external_rx_a0_nafnet_scratch.ps1"

Push-Location $RepoRoot
try {
    & $BaseScript `
        -Epochs $Epochs `
        -BatchSize $BatchSize `
        -OutputDir $OutputDir `
        -ManifestDir $ManifestDir `
        -Resume $Resume `
        -NoAutoResume:$NoAutoResume `
        -Smoke:$Smoke
} finally {
    Pop-Location
}
