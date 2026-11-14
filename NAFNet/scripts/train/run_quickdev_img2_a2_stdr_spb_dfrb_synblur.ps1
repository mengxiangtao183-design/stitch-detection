param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/QuickDevImg2_A2_STDR_SPB_DFRB_SynBlur30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_train_img2dev",
    [string]$Resume = "",
    [string]$SelectionMetric = "val_mid_psnr",
    [int]$DfrbLevels = 2,
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$BaseScript = Join-Path $PSScriptRoot "run_external_rx_a2_stdr_spb_dfrb_scratch.ps1"

& $BaseScript `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -OutputDir $OutputDir `
    -ManifestDir $ManifestDir `
    -Resume $Resume `
    -SelectionMetric $SelectionMetric `
    -DfrbLevels $DfrbLevels `
    -NoAutoResume:$NoAutoResume `
    -Smoke:$Smoke
