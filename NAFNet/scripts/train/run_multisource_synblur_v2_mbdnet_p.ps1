param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/MultiSourceV2_MBDNetP_S30_S42",
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

Push-Location $RepoRoot
try {
    $argsList = @(
        "train_nafnet_c3.py", "--arch", "mbdnet-p",
        "--data-root", "H:/DATASET", "--manifest-dir", $ManifestDir,
        "--domains", "DB", "XJ", "SD", "RX", "--output-dir", $OutputDir,
        "--patch-size", "320", "--batch-size", "$BatchSize", "--epochs", "$Epochs", "--schedule-epochs", "30",
        "--lr", "0.001", "--min-lr", "0.0000001", "--weight-decay", "0.001",
        "--width", "48", "--reconstruction-loss", "charbonnier", "--charbonnier-eps", "0.001",
        "--ssim-finetune-start", "0", "--ssim-weight", "0.0",
        "--val-every", "1", "--workers", "0", "--seed", "42", "--checkpoint-every", "5", "--log-every-steps", "25"
    )
    if ($Smoke) { $argsList += @("--max-steps", "2", "--max-val-batches", "2") }
    if ($Resume) { $argsList += @("--resume", $Resume) } elseif (!$NoAutoResume) { $argsList += "--auto-resume" }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) { throw "MultiSource SynBlur-v2 MBDNet-P training failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
