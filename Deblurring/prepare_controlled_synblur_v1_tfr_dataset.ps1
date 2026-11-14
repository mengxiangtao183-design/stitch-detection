param(
    [string]$DataRoot = "H:/DATASET",
    [string]$ManifestDir = "H:/C3Netqing/MISCFilter/manifests/c3net_controlled_synblur_v1",
    [string]$OutputRoot = "H:/DATASET/ControlledSynBlurV1_TFR",
    [switch]$Copy
)

$ErrorActionPreference = "Stop"

$PythonExe = "D:\professional_software\anaconda3\envs\c3net\python.exe"
if (!(Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Push-Location $PSScriptRoot
try {
    $argsList = @(
        "prepare_controlled_synblur_v1_tfr_dataset.py",
        "--data-root", $DataRoot,
        "--manifest-dir", $ManifestDir,
        "--output-root", $OutputRoot
    )
    if ($Copy) {
        $argsList += "--copy"
    }
    & $PythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Preparing ControlledSynBlurV1_TFR failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
