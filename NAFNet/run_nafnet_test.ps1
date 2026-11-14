param(
    [string]$Checkpoint = "checkpoints/C3NetAll_NAFNet_width32/model_best.pth",
    [string]$OutputDir = "results/C3NetAll_NAFNet_width32_midroi",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

python -u test_nafnet_c3.py `
    --data-root H:/DATASET `
    --manifest-dir H:/C3Netqing/MISCFilter/manifests/c3net_all `
    --domains DB XJ SD RX `
    --checkpoint $Checkpoint `
    --output-dir $OutputDir `
    --window-size 320 `
    --workers 0 `
    --max-images $MaxImages

if ($LASTEXITCODE -ne 0) {
    throw "test_nafnet_c3.py failed with exit code $LASTEXITCODE"
}
