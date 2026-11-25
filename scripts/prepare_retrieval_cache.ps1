param([ValidateRange(1,8)][int]$Workers = 4)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython -m arena.retrieval_cache build --workers $Workers
} else {
    & python -m arena.retrieval_cache build --workers $Workers
}
if ($LASTEXITCODE -ne 0) { throw 'Cache build did not finish. Keep the cache file and run this script again to resume.' }
