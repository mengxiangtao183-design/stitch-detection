param(
    [string]$InputDir = "H:/DATASET/quexian/dataset_96_320/synblur_original48_k3to15",
    [string]$ClearReferenceDir = "H:/DATASET/quexian/dataset_96_320/inference_original48",
    [string]$Checkpoint = "H:/C3Netqing/NAFNet/checkpoints/MultiSourceV2_MBDNetPPlus_FTDirGradSSIM5_S42/model_best.pth",
    [string]$OutputDir = "H:/C3Netqing/results/deblurring_ablation/quexian_synblur48_mbdnetpplus_restored"
)

$ErrorActionPreference = "Stop"
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run this in the c3net environment." }
if (Test-Path -LiteralPath $OutputDir) { throw "Output already exists: $OutputDir" }

python defect_detection/infer_quexian_nafnet_deblur.py `
    --input-dir $InputDir `
    --output-dir $OutputDir `
    --checkpoint $Checkpoint `
    --clear-reference-dir $ClearReferenceDir

if ($LASTEXITCODE -ne 0) { throw "Deblurring inference failed with exit code $LASTEXITCODE" }
