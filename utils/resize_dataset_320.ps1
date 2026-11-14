param(
    [string]$Root = "H:\DATASET",
    [int]$Width = 320,
    [int]$Height = 320,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path.TrimEnd("\")
if ($resolvedRoot -ne "H:\DATASET") {
    throw "Refusing to resize outside H:\DATASET. Got: $resolvedRoot"
}

Add-Type -AssemblyName System.Drawing

$imageExtensions = @(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")
$maskLikeDirs = @("gt", "edge", "edges", "mask", "masks", "label", "labels")

function Test-MaskLikePath {
    param([string]$Path)

    $parts = $Path.ToLowerInvariant() -split "[\\/]"
    foreach ($part in $parts) {
        if ($maskLikeDirs -contains $part) {
            return $true
        }
    }
    return $false
}

function Get-ImageFormatForExtension {
    param([string]$Extension)

    switch ($Extension.ToLowerInvariant()) {
        ".jpg"  { return [System.Drawing.Imaging.ImageFormat]::Jpeg }
        ".jpeg" { return [System.Drawing.Imaging.ImageFormat]::Jpeg }
        ".png"  { return [System.Drawing.Imaging.ImageFormat]::Png }
        ".bmp"  { return [System.Drawing.Imaging.ImageFormat]::Bmp }
        ".gif"  { return [System.Drawing.Imaging.ImageFormat]::Gif }
        ".tif"  { return [System.Drawing.Imaging.ImageFormat]::Tiff }
        ".tiff" { return [System.Drawing.Imaging.ImageFormat]::Tiff }
        default { throw "Unsupported image extension: $Extension" }
    }
}

function Save-ResizedBitmap {
    param(
        [System.Drawing.Bitmap]$Bitmap,
        [string]$Path,
        [System.Drawing.Imaging.ImageFormat]$Format,
        [bool]$IsJpeg
    )

    if ($IsJpeg) {
        $codec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() |
            Where-Object { $_.MimeType -eq "image/jpeg" } |
            Select-Object -First 1
        $encoderParams = New-Object System.Drawing.Imaging.EncoderParameters -ArgumentList 1
        $encoderParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter -ArgumentList (
            [System.Drawing.Imaging.Encoder]::Quality,
            [int64]95
        )
        try {
            $Bitmap.Save($Path, $codec, $encoderParams)
        }
        finally {
            $encoderParams.Dispose()
        }
    }
    else {
        $Bitmap.Save($Path, $Format)
    }
}

$files = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File |
    Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() }

$changed = 0
$skipped = 0
$failed = 0
$processed = 0
$report = New-Object System.Collections.Generic.List[object]

foreach ($file in $files) {
    $processed += 1
    $source = $null
    $dest = $null
    $graphics = $null
    $tmpPath = Join-Path $file.DirectoryName (".$($file.BaseName).resize_tmp$($file.Extension)")

    try {
        $source = [System.Drawing.Image]::FromFile($file.FullName)
        $oldWidth = $source.Width
        $oldHeight = $source.Height

        if ($oldWidth -eq $Width -and $oldHeight -eq $Height) {
            $skipped += 1
            continue
        }

        $isMaskLike = Test-MaskLikePath -Path $file.FullName
        $isJpeg = $file.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg")
        $pixelFormat = if ($isJpeg) {
            [System.Drawing.Imaging.PixelFormat]::Format24bppRgb
        }
        else {
            [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
        }

        if (-not $DryRun) {
            $dest = New-Object System.Drawing.Bitmap -ArgumentList $Width, $Height, $pixelFormat
            $dest.SetResolution($source.HorizontalResolution, $source.VerticalResolution)
            $graphics = [System.Drawing.Graphics]::FromImage($dest)

            if ($isMaskLike) {
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
                $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half
                $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::None
                $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighSpeed
            }
            else {
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
            }

            $graphics.DrawImage($source, 0, 0, $Width, $Height)
            $graphics.Dispose()
            $graphics = $null
            $source.Dispose()
            $source = $null

            $format = Get-ImageFormatForExtension -Extension $file.Extension
            Save-ResizedBitmap -Bitmap $dest -Path $tmpPath -Format $format -IsJpeg $isJpeg
            $dest.Dispose()
            $dest = $null

            Move-Item -LiteralPath $tmpPath -Destination $file.FullName -Force
        }

        $changed += 1
        $report.Add([PSCustomObject]@{
            Path = $file.FullName
            OldSize = "$oldWidth`x$oldHeight"
            NewSize = "$Width`x$Height"
            Sampler = if ($isMaskLike) { "nearest" } else { "bicubic" }
        }) | Out-Null
    }
    catch {
        $failed += 1
        Write-Warning "Failed: $($file.FullName) :: $($_.Exception.Message)"
        if (Test-Path -LiteralPath $tmpPath) {
            Remove-Item -LiteralPath $tmpPath -Force
        }
    }
    finally {
        if ($graphics -ne $null) { $graphics.Dispose() }
        if ($dest -ne $null) { $dest.Dispose() }
        if ($source -ne $null) { $source.Dispose() }
    }

    if (($processed % 500) -eq 0) {
        Write-Host "Processed $processed / $($files.Count); changed=$changed skipped=$skipped failed=$failed"
    }
}

$runsDir = Join-Path $PSScriptRoot "runs"
if (-not (Test-Path -LiteralPath $runsDir)) {
    New-Item -ItemType Directory -Path $runsDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = Join-Path $runsDir "resize_dataset_320_$timestamp.csv"
$report | Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding UTF8

Write-Host "Done. files=$($files.Count) changed=$changed skipped=$skipped failed=$failed report=$reportPath"
