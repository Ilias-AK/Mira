param(
    [string]$Checkpoint = 'runs/main/dialogue/best.pt',
    [string]$Output = 'artifacts/generator_diagnostics.json',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Project .venv not found.' }
& $venvPython -m arena.diagnostics --checkpoint $Checkpoint --device $Device --output $Output
if ($LASTEXITCODE -ne 0) { throw 'Diagnostics failed; inspect the full error above.' }
