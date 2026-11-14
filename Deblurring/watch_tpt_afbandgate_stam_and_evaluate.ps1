param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId
)

$ErrorActionPreference = 'Stop'
$projectRoot = 'H:\C3Netqing\Deblurring'
$python = 'D:\professional_software\anaconda3\python.exe'
$session = 'TPT_Net_AFBandGate_STAM_Fast100'
$modelDir = Join-Path $projectRoot "checkpoints\train\Deblurring\models\$session"
$resultDir = Join-Path $projectRoot "checkpoints\train\Deblurring\results\${session}_TestBest"
$logDir = Join-Path $resultDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$watchLog = Join-Path $logDir 'recovery_watch.log'

try {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Waiting for training PID $TrainingProcessId." |
        Out-File -FilePath $watchLog -Encoding utf8
    Wait-Process -Id $TrainingProcessId

    $bestWeights = Join-Path $modelDir 'model_best.pth'
    if (-not (Test-Path -LiteralPath $bestWeights)) {
        throw 'Training exited without model_best.pth.'
    }

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting inference." |
        Add-Content -Path $watchLog -Encoding utf8
    $inference = Start-Process -FilePath $python -ArgumentList @(
        'test.py', '--weights', $bestWeights, '--result_dir', $resultDir,
        '--datasets', 'DB', 'XJ', 'SD', 'RX', '--gpus', '0',
        '--stage1_tristripe', '--stage1_stripes', '3', '--stage1_overlap', '0.20',
        '--stage2_overlap', '0.0', '--stage2_pair_aligned',
        '--gated_csff', '--gated_csff_horizontal_kernel', '7',
        '--stitch_trajectory_alignment', '--stam_horizontal_radius', '6',
        '--stam_vertical_radius', '1', '--stam_hidden_channels', '32',
        '--frequency_refine', '--frequency_refine_hidden', '16',
        '--frequency_band_gate', '--frequency_band_low_start', '0.05',
        '--frequency_band_low_end', '0.10', '--frequency_band_high_start', '0.30',
        '--frequency_band_high_end', '0.40'
    ) -WorkingDirectory $projectRoot -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $logDir 'recovery_inference.stdout.log') `
        -RedirectStandardError (Join-Path $logDir 'recovery_inference.stderr.log')
    if ($inference.ExitCode -ne 0) {
        throw "Inference failed with exit code $($inference.ExitCode)."
    }

    function Get-EvaluationRow {
        param([string]$Label, [string]$SummaryPath)
        $summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
        $overall = $summary.summaries | Where-Object { $_.dataset -eq 'overall' }
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

    $tpt = Get-EvaluationRow 'TPT-Net' (Join-Path $projectRoot 'checkpoints\train\Deblurring\results\MPRNet_TriStripe_PairAligned_GatedCSFF_Fast100_FT_TestBest\metrics\eval_summary.json')
    $af = Get-EvaluationRow 'TPT-Net + AF-BandGate' (Join-Path $projectRoot 'checkpoints\train\Deblurring\results\TPT_Net_BandGate_Fast100_TestBest\metrics\eval_summary.json')
    $stam = Get-EvaluationRow 'TPT-Net + AF-BandGate + STAM' (Join-Path $resultDir 'metrics\eval_summary.json')
    [PSCustomObject]@{
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
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $resultDir 'metrics\comparison_vs_tpt_afbandgate.json') -Encoding utf8

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Inference and comparison completed." |
        Add-Content -Path $watchLog -Encoding utf8
}
catch {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR: $($_.Exception.Message)" |
        Add-Content -Path $watchLog -Encoding utf8
    exit 1
}
