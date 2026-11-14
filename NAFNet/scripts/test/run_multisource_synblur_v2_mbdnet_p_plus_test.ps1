param(
    [string]$Checkpoint = "checkpoints/MultiSourceV2_MBDNetPPlus_S30_S42/model_best.pth",
    [string]$OutputDir = "results/MultiSourceV2_MBDNetPPlus_S30_S42_midroi",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2",
    [int]$MaxImages = 0,
    [switch]$FastFP16
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
if (!(Test-Path -LiteralPath $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
foreach ($domain in @("DB", "XJ", "SD", "RX", "RXHF", "TJML")) {
    if (!(Test-Path -LiteralPath (Join-Path $ManifestDir "test_$domain.tsv"))) { throw "Missing test_$domain.tsv under $ManifestDir" }
}

Push-Location $RepoRoot
try {
    $argsList = @("test_nafnet_c3.py", "--arch", "mbdnet-p-plus",
        "--data-root", "H:/DATASET", "--manifest-dir", $ManifestDir, "--domains", "DB", "XJ", "SD", "RX", "RXHF", "TJML",
        "--checkpoint", $Checkpoint, "--output-dir", $OutputDir, "--window-size", "320", "--workers", "0", "--max-images", "$MaxImages",
        "--width", "64")
    if ($FastFP16) { $argsList += @("--amp-inference", "--channels-last") }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) { throw "MultiSource SynBlur-v2 MBDNet-P+ test failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
