param(
    [string]$Checkpoint = "checkpoints/Ablation_A0_NAFNet_FT15_S42/model_best.pth",
    [string]$OutputDir = "results/Ablation_A0_NAFNet_FT15_S42_midroi",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"

& "$PSScriptRoot\run_nafnet_test.ps1" `
    -Checkpoint $Checkpoint `
    -OutputDir $OutputDir `
    -MaxImages $MaxImages
