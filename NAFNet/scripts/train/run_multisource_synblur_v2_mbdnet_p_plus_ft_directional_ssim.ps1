param(
    [int]$Epochs = 5,
    [int]$BatchSize = 2,
    [string]$InitCheckpoint = "checkpoints/MultiSourceV2_MBDNetPPlus_FTPSNR10_S42/model_best.pth",
    [string]$OutputDir = "checkpoints/MultiSourceV2_MBDNetPPlus_FTDirGradSSIM5_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2",
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
foreach ($required in @("train.tsv", "val.tsv")) {
    if (!(Test-Path -LiteralPath (Join-Path $ManifestDir $required))) { throw "Missing $required under $ManifestDir" }
}
if (!$Resume -and !(Test-Path -LiteralPath $InitCheckpoint)) { throw "Initial checkpoint not found: $InitCheckpoint" }

Push-Location $RepoRoot
try {
    $argsList = @(
        "train_nafnet_c3.py", "--arch", "mbdnet-p-plus",
        "--data-root", "H:/DATASET", "--manifest-dir", $ManifestDir,
        "--domains", "DB", "XJ", "SD", "RX", "--output-dir", $OutputDir,
        "--patch-size", "320", "--batch-size", "$BatchSize", "--epochs", "$Epochs", "--schedule-epochs", "$Epochs",
        "--lr", "0.000005", "--min-lr", "0.0000001", "--weight-decay", "0.001",
        "--width", "64", "--reconstruction-loss", "psnr",
        "--horizontal-gradient-weight", "0.02", "--gradient-eps", "0.001",
        "--ssim-finetune-start", "1", "--ssim-weight", "0.03",
        "--amp", "--grad-accum-steps", "2",
        "--val-every", "1", "--workers", "0", "--seed", "42", "--checkpoint-every", "5", "--log-every-steps", "25"
    )
    if ($Smoke) { $argsList += @("--max-steps", "2", "--max-val-batches", "2") }
    if ($Resume) { $argsList += @("--resume", $Resume) } else { $argsList += @("--init-checkpoint", $InitCheckpoint) }
    if (!$Resume -and !$NoAutoResume) {
        Write-Output "Fine-tuning from $InitCheckpoint with a fresh optimizer and scheduler."
    }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) { throw "MBDNet-P+ directional-gradient + SSIM fine-tuning failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
