param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('mbdnet_p_plus', 'mprnet', 'nafnet', 'restormer', 'stripformer', 'banet', 'loformer', 'mimo_unet_plus', 'hinet', 'ufpnet', 'sfnet')]
    [string]$Model,
    [int]$Epochs = 30,
    [int]$BatchSize = 0,
    [int]$GradAccumSteps = 0,
    [string]$OutputDir = '',
    [string]$ManifestDir = 'H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2',
    [string]$Resume = '',
    [switch]$NoAutoResume,
    [switch]$NoAmp,
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
if ($env:CONDA_DEFAULT_ENV -ne 'c3net') {
    throw 'Activate the c3net Conda environment before running this script.'
}

$SuiteRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $SuiteRoot
$Arguments = @(
    'train_comparison.py', '--model', $Model,
    '--data-root', 'H:/DATASET',
    '--manifest-dir', $ManifestDir,
    '--epochs', $Epochs,
    '--schedule-epochs', '30',
    '--patch-size', '320',
    '--workers', '0', '--seed', '42',
    '--val-every', '1', '--checkpoint-every', '5', '--log-every-steps', '25'
)
if ($BatchSize -gt 0) { $Arguments += @('--batch-size', $BatchSize) }
if ($GradAccumSteps -gt 0) { $Arguments += @('--grad-accum-steps', $GradAccumSteps) }
if ($OutputDir) { $Arguments += @('--output-dir', $OutputDir) }
if ($Resume) { $Arguments += @('--resume', $Resume) }
elseif (-not $NoAutoResume) { $Arguments += '--auto-resume' }
# The official BANet and Stripformer settings use FP32 Adam at 1e-4.
# Their entries in model_registry.json provide this automatically.
$Fp32Models = @('banet', 'stripformer', 'mimo_unet_plus', 'hinet', 'ufpnet', 'sfnet')
if (-not $NoAmp -and $Model -notin $Fp32Models) { $Arguments += '--amp' }
if ($Smoke) { $Arguments += @('--max-steps', '2', '--max-val-batches', '2') }

& python -u @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Unified deblurring comparison training failed with exit code $LASTEXITCODE"
}
