param(
    [int]$Epochs = 20,
    [int]$BatchSize = 4,
    [string]$InitCheckpoint = "checkpoints/LineTraceV3_A0_NAFNet_Pre30_S42/model_best.pth",
    [string]$OutputDir = "checkpoints/LineTraceSynBlurV2_A0_FromV3FT20_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_controlled_linetrace_synblur_v2",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
if (!(Test-Path -LiteralPath $InitCheckpoint)) { throw "Initial V3 checkpoint not found: $InitCheckpoint" }
foreach ($required in @("train.tsv", "val.tsv")) {
    if (!(Test-Path -LiteralPath (Join-Path $ManifestDir $required))) { throw "Missing $required under $ManifestDir" }
}
Push-Location $RepoRoot
try {
    $argsList = @(
        "train_nafnet_c3.py", "--arch", "nafnet", "--init-checkpoint", $InitCheckpoint,
        "--data-root", "H:/DATASET", "--manifest-dir", $ManifestDir, "--domains", "DB", "XJ", "SD", "--output-dir", $OutputDir,
        "--patch-size", "320", "--batch-size", "$BatchSize", "--epochs", "$Epochs", "--schedule-epochs", "$Epochs",
        "--lr", "0.0005", "--min-lr", "0.0000001", "--weight-decay", "0.001",
        "--width", "32", "--enc-blks", "1", "1", "1", "28", "--middle-blks", "1", "--dec-blks", "1", "1", "1", "1",
        "--val-every", "5", "--workers", "0", "--seed", "42", "--checkpoint-every", "5", "--log-every-steps", "25"
    )
    if ($Smoke) { $argsList += @("--max-steps", "2", "--max-val-batches", "2") }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) { throw "LineTrace V2 A0 fine-tuning from V3 failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
