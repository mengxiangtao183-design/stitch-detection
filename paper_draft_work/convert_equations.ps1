param(
  [Parameter(Mandatory=$true)][string]$InputDocx,
  [Parameter(Mandatory=$true)][string]$EquationJson
)

$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
  $doc = $word.Documents.Open($InputDocx)
  # Consume a Word equation-engine first-use quirk with a disposable placeholder.
  $warmRange = $doc.Content.Duplicate
  $warmFind = $warmRange.Find
  $warmFind.Text = '[[EQWARM]]'
  $warmFind.Forward = $true
  $warmFind.Wrap = 0
  if ($warmFind.Execute()) {
    $warmRange.Text = 'x=1'
    $null = $doc.OMaths.Add($warmRange)
  }
  $items = Get-Content -LiteralPath $EquationJson -Raw -Encoding UTF8 | ConvertFrom-Json
  foreach ($item in $items) {
    $rng = $doc.Content.Duplicate
    $find = $rng.Find
    $find.ClearFormatting()
    $find.Text = [string]$item.placeholder
    $find.Forward = $true
    $find.Wrap = 0
    if (-not $find.Execute()) {
      throw "Equation placeholder not found: $($item.placeholder)"
    }
    $rng.Text = [string]$item.formula
    $mathRange = $rng.Duplicate
    $null = $doc.OMaths.Add($mathRange)
    if ($mathRange.OMaths.Count -gt 0) {
      $mathRange.OMaths.Item(1).BuildUp()
    }
  }
  for ($i = $doc.OMaths.Count; $i -ge 1; $i--) {
    $om = $doc.OMaths.Item($i)
    if ([string]$om.Range.Text -like '*在此处键入公式*') {
      $om.Range.Delete() | Out-Null
    }
  }
  $doc.Save()
  Write-Output "Converted $($items.Count) equations in $InputDocx"
}
finally {
  if ($doc -ne $null) {
    try { $doc.Close($false) } catch {}
  }
  try { $word.Quit() } catch {}
  if ($doc -ne $null) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null }
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
  [GC]::Collect()
  [GC]::WaitForPendingFinalizers()
}
