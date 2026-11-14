param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Seeds = @(42)
$M2Config = Join-Path $ProjectRoot "../_archive/deblurring_nonpaper_yaml_2026-11-14/training_mprnet_m2_generalization_pair_fast100.yml"
$M5Config = Join-Path $ProjectRoot "../_archive/deblurring_nonpaper_yaml_2026-11-14/training_mprnet_m5v3_generalization_pair_fast100.yml"

Write-Host "Paired generalization runner v2: 2 runs, seed 42 only"
Write-Host "Sessions: MPRNet_M2_GeneralizationPairSafeTensor_Seed42, MPRNet_M5v3_GeneralizationPairSafeTensor_Seed42"
Write-Host "Checkpoint mode: safetensors weights with deterministic epoch resume"

function Invoke-PairedTraining {
    param(
        [string]$Config,
        [int]$Seed,
        [string]$Session,
        [string]$Label,
        [int]$MaxAttempts = 3
    )

    for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
        & $Python (Join-Path $ProjectRoot "train.py") `
            --config $Config `
            --seed $Seed `
            --session $Session
        if ($LASTEXITCODE -eq 0) {
            return
        }

        Write-Warning (
            "$Label exited with code $LASTEXITCODE for seed $Seed " +
            "(attempt $Attempt/$MaxAttempts). The next attempt will resume " +
            "from the latest valid checkpoint."
        )
        if ($Attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds 3
        }
    }

    throw "$Label paired run failed for seed $Seed after $MaxAttempts attempts"
}

foreach ($Seed in $Seeds) {
    Invoke-PairedTraining `
        -Config $M2Config `
        -Seed $Seed `
        -Session "MPRNet_M2_GeneralizationPairSafeTensor_Seed$Seed" `
        -Label "M2"

    Invoke-PairedTraining `
        -Config $M5Config `
        -Seed $Seed `
        -Session "MPRNet_M5v3_GeneralizationPairSafeTensor_Seed$Seed" `
        -Label "M5-v3"
}
