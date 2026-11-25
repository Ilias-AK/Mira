param(
    [string]$Checkpoint = '',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [ValidateRange(0,96)][int]$NoRepeatNgramSize = 0,
    [switch]$RepeatGuard
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Run scripts/bootstrap_lenovo.ps1 first.' }
$chatArgs = @('-m','arena.cli','--device',$Device)
if ($Checkpoint) { $chatArgs += @('--checkpoint',$Checkpoint) }
if ($RepeatGuard) { $chatArgs += '--repeat-guard' }
if ($NoRepeatNgramSize) { $chatArgs += @('--no-repeat-ngram-size',"$NoRepeatNgramSize") }
& $venvPython @chatArgs
if ($LASTEXITCODE -ne 0) { throw 'Chat process failed.' }
