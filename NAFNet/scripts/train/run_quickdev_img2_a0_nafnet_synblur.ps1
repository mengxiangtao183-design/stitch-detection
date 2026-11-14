param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/QuickDevImg2_A0_NAFNet_SynBlur30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_train_img2dev",
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$BaseScript = Join-Path $PSScriptRoot "run_external_rx_a0_nafnet_scratch.ps1"

& $BaseScript `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -OutputDir $OutputDir `
    -ManifestDir $ManifestDir `
    -Resume $Resume `
    -NoAutoResume:$NoAutoResume `
    -Smoke:$Smoke
