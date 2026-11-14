param(
    [int]$DelaySeconds = 3600
)

$ErrorActionPreference = 'Stop'

$projectRoot = 'H:\C3Netqing\Deblurring'
$python = 'D:\professional_software\anaconda3\python.exe'
$config = '../_archive/deblurring_nonpaper_yaml_2026-11-14/training_tpt_net_stam_fast100.yml'
$session = 'TPT_Net_AFBandGate_STAM_Fast100'
$modelDir = Join-Path $projectRoot "checkpoints\train\Deblurring\models\$session"
$resultDir = Join-Path $projectRoot "checkpoints\train\Deblurring\results\${session}_TestBest"
$logDir = Join-Path $resultDir 'logs'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$scheduleLog = Join-Path $logDir 'scheduled_run.log'
$trainLog = Join-Path $logDir 'train.stdout.log'
$trainErrorLog = Join-Path $logDir 'train.stderr.log'
$inferLog = Join-Path $logDir 'inference.stdout.log'
$inferErrorLog = Join-Path $logDir 'inference.stderr.log'

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Scheduled; waiting $DelaySeconds seconds." |
    Out-File -FilePath $scheduleLog -Encoding utf8
Start-Sleep -Seconds $DelaySeconds

Set-Location $projectRoot
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting training." |
    Add-Content -Path $scheduleLog -Encoding utf8

& $python train.py --config $config 1> $trainLog 2> $trainErrorLog
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training failed with exit code $LASTEXITCODE." |
        Add-Content -Path $scheduleLog -Encoding utf8
    exit $LASTEXITCODE
}

$bestWeights = Join-Path $modelDir 'model_best.pth'
if (-not (Test-Path -LiteralPath $bestWeights)) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training completed but model_best.pth is missing." |
        Add-Content -Path $scheduleLog -Encoding utf8
    exit 2
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting 800-image inference with model_best.pth." |
    Add-Content -Path $scheduleLog -Encoding utf8

& $python test.py `
    --weights $bestWeights `
    --result_dir $resultDir `
    --datasets DB XJ SD RX `
    --gpus 0 `
    --stage1_tristripe `
    --stage1_stripes 3 `
    --stage1_overlap 0.20 `
    --stage2_overlap 0.0 `
    --stage2_pair_aligned `
    --gated_csff `
    --gated_csff_horizontal_kernel 7 `
    --stitch_trajectory_alignment `
    --stam_horizontal_radius 6 `
    --stam_vertical_radius 1 `
    --stam_hidden_channels 32 `
    --frequency_refine `
    --frequency_refine_hidden 16 `
    --frequency_band_gate `
    --frequency_band_low_start 0.05 `
    --frequency_band_low_end 0.10 `
    --frequency_band_high_start 0.30 `
    --frequency_band_high_end 0.40 `
    1> $inferLog 2> $inferErrorLog

if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Inference failed with exit code $LASTEXITCODE." |
        Add-Content -Path $scheduleLog -Encoding utf8
    exit $LASTEXITCODE
}

function Get-EvaluationRow {
    param(
        [string]$Label,
        [string]$SummaryPath
    )

    $summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $overall = $summary.summaries | Where-Object { $_.dataset -eq 'overall' }
    if ($null -eq $overall) {
        throw "Overall metrics are missing: $SummaryPath"
    }
    [PSCustomObject]@{
        model = $Label
        full_psnr_macro = [double]$overall.restored_psnr
        middle_stripe_psnr_macro = [double]$overall.restored_roi_psnr
        full_ssim_macro = [double]$overall.restored_ssim
        middle_stripe_ssim_macro = [double]$overall.restored_roi_ssim
        full_psnr_sum_4datasets = [double]$overall.restored_psnr * 4.0
        middle_stripe_psnr_sum_4datasets = [double]$overall.restored_roi_psnr * 4.0
    }
}

$tptSummary = Join-Path $projectRoot 'checkpoints\train\Deblurring\results\MPRNet_TriStripe_PairAligned_GatedCSFF_Fast100_FT_TestBest\metrics\eval_summary.json'
$afSummary = Join-Path $projectRoot 'checkpoints\train\Deblurring\results\TPT_Net_BandGate_Fast100_TestBest\metrics\eval_summary.json'
$stamSummary = Join-Path $resultDir 'metrics\eval_summary.json'

$tpt = Get-EvaluationRow -Label 'TPT-Net' -SummaryPath $tptSummary
$af = Get-EvaluationRow -Label 'TPT-Net + AF-BandGate' -SummaryPath $afSummary
$stam = Get-EvaluationRow -Label 'TPT-Net + AF-BandGate + STAM' -SummaryPath $stamSummary
$comparison = [PSCustomObject]@{
    aggregation = 'per-dataset image means, then equal-weight macro mean across DB/XJ/SD/RX'
    middle_stripe = 'Stage-1 middle-stripe non-overlapping core'
    results = @($tpt, $af, $stam)
    stam_delta_vs_af_bandgate = [PSCustomObject]@{
        full_psnr_macro = $stam.full_psnr_macro - $af.full_psnr_macro
        middle_stripe_psnr_macro = $stam.middle_stripe_psnr_macro - $af.middle_stripe_psnr_macro
        full_ssim_macro = $stam.full_ssim_macro - $af.full_ssim_macro
        middle_stripe_ssim_macro = $stam.middle_stripe_ssim_macro - $af.middle_stripe_ssim_macro
        full_psnr_sum_4datasets = $stam.full_psnr_sum_4datasets - $af.full_psnr_sum_4datasets
        middle_stripe_psnr_sum_4datasets = $stam.middle_stripe_psnr_sum_4datasets - $af.middle_stripe_psnr_sum_4datasets
    }
}
$comparisonPath = Join-Path $resultDir 'metrics\comparison_vs_tpt_afbandgate.json'
$comparison | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $comparisonPath -Encoding utf8

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training and inference completed successfully." |
    Add-Content -Path $scheduleLog -Encoding utf8
