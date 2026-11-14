param(
    [int]$Epochs = 30,
    [int]$BatchSize = 2,
    [string]$Session = "MultiSourceV2_MPRNetOriginal_S30_S42",
    [switch]$NoAutoResume
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"
$RepoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Config = Join-Path $RepoRoot "training_mprnet_original_multisource_synblur_v2_s30.yml"
$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2"
if ($env:CONDA_DEFAULT_ENV -ne "c3net") { throw "Run: conda activate c3net" }
foreach ($required in @("train.tsv", "val.tsv")) {
    if (!(Test-Path -LiteralPath (Join-Path $ManifestDir $required))) { throw "Missing $required under $ManifestDir" }
}
if (!(Test-Path -LiteralPath $Config)) { throw "Config not found: $Config" }

Push-Location $RepoRoot
try {
    $argsList = @(
        "train.py", "--config", $Config, "--seed", "42", "--session", $Session,
        "--epochs", "$Epochs", "--batch-size", "$BatchSize"
    )
    if ($NoAutoResume) { $argsList += "--no-auto-resume" }
    python -u @argsList
    if ($LASTEXITCODE -ne 0) { throw "Original MPRNet MultiSource SynBlur-v2 training failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
