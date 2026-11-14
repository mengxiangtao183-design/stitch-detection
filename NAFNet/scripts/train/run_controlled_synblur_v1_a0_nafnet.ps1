param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/ControlledSynBlurV1_A0_NAFNet_S30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_controlled_synblur_v1",
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
