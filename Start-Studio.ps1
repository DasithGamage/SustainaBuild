param([string]$Python = "", [int]$Port = 8000)
$ErrorActionPreference = "Stop"
if (-not $Python) {
    $reviewPython = Join-Path $env:TEMP "sb-review-env\Scripts\python.exe"
    $localPython = Join-Path $env:TEMP "sb-env\Scripts\python.exe"
    if (Test-Path -LiteralPath $reviewPython) { $Python = $reviewPython }
    elseif (Test-Path -LiteralPath $localPython) { $Python = $localPython }
    else { $Python = "python" }
}
Write-Host "SustainaBuild Studio: http://127.0.0.1:$Port"
Write-Host "Keep this terminal open. Press Ctrl+C to stop."
& $Python (Join-Path $PSScriptRoot "sustainabuild_web.py") --port $Port
