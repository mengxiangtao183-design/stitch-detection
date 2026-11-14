param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('mbdnet_p_plus', 'mprnet', 'nafnet', 'restormer', 'stripformer', 'banet', 'loformer', 'mimo_unet_plus', 'hinet', 'ufpnet', 'sfnet')]
    [string]$Model,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$OutputDir = '',
    [string]$ManifestDir = 'H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2',
    [int]$MaxImages = 0,
    [switch]$Amp,
    [switch]$SaveImages
)

$ErrorActionPreference = 'Stop'
if ($env:CONDA_DEFAULT_ENV -ne 'c3net') {
    throw 'Activate the c3net Conda environment before running this script.'
}

$SuiteRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $SuiteRoot
$Arguments = @(
    'test_comparison.py', '--model', $Model,
    '--checkpoint', $Checkpoint,
    '--data-root', 'H:/DATASET',
    '--manifest-dir', $ManifestDir,
    '--domains', 'DB', 'XJ', 'SD', 'RX', 'RXHF', 'TJML',
    '--window-size', '320', '--workers', '0'
)
if ($OutputDir) { $Arguments += @('--output-dir', $OutputDir) }
if ($MaxImages -gt 0) { $Arguments += @('--max-images', $MaxImages) }
if ($Amp) { $Arguments += '--amp' }
if ($SaveImages) { $Arguments += '--save-images' }

& python -u @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Unified deblurring comparison test failed with exit code $LASTEXITCODE"
}
