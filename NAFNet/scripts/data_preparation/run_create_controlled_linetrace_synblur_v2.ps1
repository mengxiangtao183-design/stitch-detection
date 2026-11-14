param(
    [string]$DataRoot = "H:\DATASET",
    [string]$BaseManifestDir = "H:\C3Netqing\MISCFilter\manifests\c3net_external_rx",
    [string]$ManifestDir = "H:\C3Netqing\MISCFilter\manifests\c3net_controlled_linetrace_synblur_v2",
    [string]$OutputSubdir = "ImgsSynLineV2",
    [int]$Seed = 42,
    [switch]$Overwrite,
    [switch]$WritePreview
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $scriptDir "create_controlled_linetrace_synblur_v2.py"

$argsList = @(
    $py,
    "--data-root", $DataRoot,
    "--base-manifest-dir", $BaseManifestDir,
    "--manifest-dir", $ManifestDir,
    "--output-subdir", $OutputSubdir,
    "--seed", "$Seed"
)

if ($Overwrite) {
    $argsList += "--overwrite"
}
if ($WritePreview) {
    $argsList += "--write-preview"
}

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "create_controlled_linetrace_synblur_v2.py failed with exit code $LASTEXITCODE"
}
