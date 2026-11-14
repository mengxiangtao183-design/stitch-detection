param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Seed = 42
$M2Config = Join-Path $ProjectRoot "../_archive/deblurring_nonpaper_yaml_2026-11-14/training_mprnet_m2_generalization_pair_fast100.yml"
$M5Config = Join-Path $ProjectRoot "../_archive/deblurring_nonpaper_yaml_2026-11-14/training_mprnet_m5v3_generalization_pair_fast100.yml"
$M2Session = "MPRNet_M2_GeneralizationPairSafeTensor_Seed42"
$M5Session = "MPRNet_M5v3_GeneralizationPairSafeTensor_Seed42"

Write-Host "Paired generalization runner v2: 2 runs, seed 42 only"
Write-Host "M2 session: $M2Session"
Write-Host "M5 session: $M5Session"
Write-Host "Checkpoint mode: safetensors weights with deterministic epoch resume"

function Invoke-StableTraining {
    param(
        [string]$Config,
        [string]$Session,
        [string]$Label,
        [int]$MaxAttempts = 3
    )

    for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
        Write-Host "Starting $Label attempt $Attempt/$MaxAttempts"
        & $Python (Join-Path $ProjectRoot "train.py") `
            --config $Config `
            --seed $Seed `
            --session $Session
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$Label completed"
            return
        }

        Write-Warning (
            "$Label exited with code $LASTEXITCODE on attempt " +
            "$Attempt/$MaxAttempts. Resuming from the latest valid checkpoint."
        )
        if ($Attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds 3
        }
    }

    throw "$Label failed after $MaxAttempts attempts"
}

Invoke-StableTraining -Config $M2Config -Session $M2Session -Label "M2"
Invoke-StableTraining -Config $M5Config -Session $M5Session -Label "M5-v3"
