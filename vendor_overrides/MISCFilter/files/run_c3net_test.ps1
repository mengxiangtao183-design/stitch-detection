param(
    [string]$Checkpoint = "checkpoints/C3NetAll_MISCFilters/model_latest.pth",
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONFAULTHANDLER = "1"

if ($env:CONDA_DEFAULT_ENV -ne "c3net") {
    throw "Current conda env is '$env:CONDA_DEFAULT_ENV'. Expected 'c3net'. Run: conda activate c3net"
}

python -u test_c3net.py `
    --data-root H:/DATASET `
    --domains DB XJ SD RX `
    --checkpoint $Checkpoint `
    --window-size 320 `
    --workers 0 `
    --max-images $MaxImages

if ($LASTEXITCODE -ne 0) {
    throw "test_c3net.py failed with exit code $LASTEXITCODE"
}
