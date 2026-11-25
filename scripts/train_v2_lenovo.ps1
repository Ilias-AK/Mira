param(
    [string]$RunName = 'training_v2',
    [string]$Config = 'configs/training_v2.json',
    [string]$PythonPath = '',
    [switch]$Resume,
    [switch]$CollectOnly,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($PythonPath) { $python = $PythonPath }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Project Python is missing. Use the existing Lenovo virtual environment.'
}
$arguments = @('-X','utf8','-m','arena.next_training','--config',$Config,'--run-name',$RunName)
if ($Resume) { $arguments += '--resume' }
if ($CollectOnly) { $arguments += '--collect-only' }
if ($CheckOnly) { $arguments += '--check-only' }
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Training did not finish. Keep runs/$RunName and resume with identical settings after resolving the reported error."
}
if ($CheckOnly) {
    Write-Host 'Inputs are ready. This check did not start training.'
} else {
    Write-Host "Reports: runs/$RunName/training_v2_results.zip. Keep the model weights on this laptop."
}
