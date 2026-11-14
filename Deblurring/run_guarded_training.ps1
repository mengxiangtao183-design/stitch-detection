param(
    [string]$Config = '../_archive/deblurring_nonpaper_yaml_2026-11-14/training_tptrnet_tcp_scratch_full150.yml',
    [string]$PythonExe = 'D:\professional_software\anaconda3\envs\c3net\python.exe',
    [int]$MaxRestarts = 40,
    [int]$RestartDelaySeconds = 10
)

# Keep the console visible: each child run writes normal training output here.
# train.py uses RESUME=true and safetensors checkpoints, so a native Python or
# CUDA-process exit restarts from the most recently completed epoch.
$ErrorActionPreference = 'Continue'
$configPath = Join-Path $PSScriptRoot $Config
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Training config not found: $configPath"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Push-Location $PSScriptRoot
try {
    & $PythonExe -c "import torch; print('Guard Python:', __import__('sys').executable); print('PyTorch:', torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "The selected Python cannot import PyTorch: $PythonExe"
    }

    for ($attempt = 0; $attempt -le $MaxRestarts; $attempt++) {
        $started = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Host "[$started] Starting training process (restart $attempt/$MaxRestarts)..." -ForegroundColor Cyan

        & $PythonExe (Join-Path $PSScriptRoot 'train.py') --config $configPath
        $exitCode = $LASTEXITCODE

        $epochProbe = "from pathlib import Path; import sys; from config import Config; from utils import load_start_epoch; c=Config(sys.argv[1]); p=Path(c.TRAINING.SAVE_DIR)/c.MODEL.MODE/'models'/c.MODEL.SESSION/'model_latest.pth'; print(load_start_epoch(str(p)) if p.exists() else 0)"
        $completedEpoch = [int](& $PythonExe -c $epochProbe $configPath)
        $targetEpoch = [int](& $PythonExe -c "from config import Config; import sys; print(Config(sys.argv[1]).OPTIM.NUM_EPOCHS)" $configPath)

        if ($exitCode -eq 0 -and $completedEpoch -ge $targetEpoch) {
            Write-Host "Training completed normally at epoch $completedEpoch." -ForegroundColor Green
            exit 0
        }

        if ($attempt -eq $MaxRestarts) {
            Write-Host "Training exited with code $exitCode after the maximum restart count." -ForegroundColor Red
            exit $exitCode
        }

        Write-Warning "Training exited with code $exitCode at saved epoch $completedEpoch/$targetEpoch. Checkpoints are retained; restarting in $RestartDelaySeconds seconds."
        Start-Sleep -Seconds $RestartDelaySeconds
    }
}
finally {
    Pop-Location
}
