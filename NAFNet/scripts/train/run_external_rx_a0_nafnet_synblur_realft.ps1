param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/ExternalRX_A0_NAFNet_SynBlurV1SplitValPre30_RealFT30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_external_rx",
    [string]$InitCheckpoint = "checkpoints/ExternalRX_A0_NAFNet_SynBlurV1SplitValPre30_S42/model_best.pth",
    [string]$Resume = "",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$BaseScript = Join-Path $PSScriptRoot "run_external_rx_a0_nafnet_scratch.ps1"

Push-Location $RepoRoot
try {
    if (($Resume -eq "") -and !(Test-Path -LiteralPath $InitCheckpoint)) {
        throw "Missing A0 SynBlur pretrain checkpoint: $InitCheckpoint"
    }
    & $BaseScript `
        -Epochs $Epochs `
        -BatchSize $BatchSize `
        -OutputDir $OutputDir `
        -ManifestDir $ManifestDir `
        -Resume $Resume `
        -InitCheckpoint $InitCheckpoint `
        -NoAutoResume `
        -Smoke:$Smoke
} finally {
    Pop-Location
}
