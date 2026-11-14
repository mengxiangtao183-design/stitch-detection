param(
    [string]$Config = "../_archive/deblurring_nonpaper_yaml_2026-11-14/training_tfr_net_v2_lite_controlled_synblur_v1_fast15_nokd.yml",
    [int]$MaxRestarts = 20,
    [int]$RestartDelaySeconds = 10
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "run_guarded_training.ps1") `
    -Config $Config `
    -MaxRestarts $MaxRestarts `
    -RestartDelaySeconds $RestartDelaySeconds
