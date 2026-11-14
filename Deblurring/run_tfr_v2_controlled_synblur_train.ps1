param(
    [string]$Config = "training_tfr_net_v2_controlled_synblur_v1_s30_nokd.yml",
    [int]$MaxRestarts = 40,
    [int]$RestartDelaySeconds = 10
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "run_guarded_training.ps1") `
    -Config $Config `
    -MaxRestarts $MaxRestarts `
    -RestartDelaySeconds $RestartDelaySeconds
