param(
    [string]$Weights = "H:/C3Netqing/Deblurring/checkpoints/train/Deblurring/models/TFR_Net_v2_ControlledSynBlurV1_S30_NoKD/model_best.pth",
    [string]$InputDir = "H:/DATASET/ControlledSynBlurV1_TFR",
    [string]$ResultDir = "H:/C3Netqing/Deblurring/checkpoints/train/Deblurring/results/TFR_Net_v2_ControlledSynBlurV1_S30_NoKD_TestBest/restored",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"

$PythonExe = "D:\professional_software\anaconda3\envs\c3net\python.exe"
if (!(Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Push-Location $PSScriptRoot
try {
    & $PythonExe test.py `
        --input_dir $InputDir `
        --result_dir $ResultDir `
        --weights $Weights `
        --datasets DB XJ SD RX `
        --split test `
        --input_subdir input `
        --clear_subdir clear `
        --metrics_dir (Join-Path $ResultDir "metrics") `
        --num_workers 0 `
        --max_images $MaxImages `
        --n_feat 96 `
        --scale_unetfeats 48 `
        --scale_orsnetfeats 32 `
        --num_cab 8 `
        --stage1_tristripe `
        --stage1_stripes 3 `
        --stage1_overlap 0.20 `
        --stage2_pair_aligned `
        --tristripe_context_pyramid `
        --tristripe_context_blocks 3 `
        --horizontal_trajectory_blocks `
        --tfr_net_core `
        --tfr_blocks 8 `
        --tfr_horizontal_kernel 15 `
        --tfr_vertical_kernel 5 `
        --tfr_refiner_steps 3 `
        --frequency_refine `
        --frequency_refine_hidden 16 `
        --frequency_band_gate `
        --frequency_band_low_start 0.05 `
        --frequency_band_low_end 0.10 `
        --frequency_band_high_start 0.30 `
        --frequency_band_high_end 0.40

    if ($LASTEXITCODE -ne 0) {
        throw "TFR-v2 Controlled-SynBlur test failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
