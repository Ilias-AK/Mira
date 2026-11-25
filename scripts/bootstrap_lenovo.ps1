param(
    [switch]$Cpu,
    [ValidateSet('cu126','cu128')][string]$Cuda = 'cu126'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv (Join-Path $projectRoot '.venv')
    } else {
        & python -m venv (Join-Path $projectRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Creating Python environment failed. Install Python 3.11 or newer.' }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
$wheelIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { "https://download.pytorch.org/whl/$Cuda" }
$torchBuild = if ($Cpu) { 'cpu' } else { $Cuda }
& $venvPython -m pip install "torch==2.10.0+$torchBuild" --index-url $wheelIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $venvPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $venvPython -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
if ($Cpu) { & $venvPython -m arena.doctor } else { & $venvPython -m arena.doctor --require-cuda }
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed. Inspect the diagnostic above.' }
Write-Host 'Environment ready. Python: ' $venvPython
