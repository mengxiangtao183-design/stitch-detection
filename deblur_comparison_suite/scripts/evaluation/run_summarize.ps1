param(
    [string]$ResultsRoot = '',
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$SuiteRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $SuiteRoot
$Arguments = @('summarize_results.py')
if ($ResultsRoot) { $Arguments += @('--results-root', $ResultsRoot) }
if ($Output) { $Arguments += @('--output', $Output) }
& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Unified result aggregation failed with exit code $LASTEXITCODE"
}
