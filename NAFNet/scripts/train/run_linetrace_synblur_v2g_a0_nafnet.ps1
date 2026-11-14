param(
    [int]$Epochs = 30,
    [int]$BatchSize = 4,
    [string]$OutputDir = "checkpoints/LineTraceSynBlurV2G_A0_NAFNet_S30_S42",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_controlled_linetrace_synblur_v2",
    [string]$Resume = "",
    [switch]$NoAutoResume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
if (!(Test-Path -LiteralPath (Join-Path $ManifestDir "train.tsv"))) { throw "Missing manifests under $ManifestDir" }
Push-Location $RepoRoot
try {
    $argsList = @(
        "train_nafnet_c3.py", "--arch", "nafnet",
        "--data-root", "H:/DATASET", "--manifest-dir", $ManifestDir,
        "--domains", "DB", "XJ", "SD", "--output-dir", $OutputDir,
        "--patch-size", "320", "--batch-size", "$BatchSize", "--epochs", "$Epochs",
        "--lr", "0.001", "--min-lr", "0.0000001", "--weight-decay", "0.001",
        "--width", "32", "--enc-blks", "1", "1", "1", "28", "--middle-blks", "1", "--dec-blks", "1", "1", "1", "1",
        "--source-style-prob", "0.60", "--source-gain", "0.08", "--source-bias", "0.03",
        "--source-gamma", "0.10", "--source-noise-std", "0.010", "--source-band-gain", "0.06",
        "--val-every", "5", "--workers", "0", "--seed", "42", "--checkpoint-every", "5", "--log-every-steps", "25"
    )
    if ($Smoke) { $argsList += @("--max-steps", "2", "--max-val-batches", "4") }
    if ($Resume) { $argsList += @("--resume", $Resume) } elseif (!$NoAutoResume) { $argsList += "--auto-resume" }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) { throw "LineTrace SynBlur-v2-G A0 NAFNet training failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
