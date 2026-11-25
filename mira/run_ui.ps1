# Локальный веб-интерфейс Mira. По умолчанию — режим конкурса.
param(
    [int]$Port = 8765,
    [switch]$Chat,
    [switch]$Typing,
    [switch]$Retrieval,
    [switch]$NoIntent,
    [switch]$NoTypos,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root

chcp 65001 > $null
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$here;$root"
$env:PYTHONIOENCODING = "utf-8"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Не найден Python. Установите Python 3.11 и зависимости из README.md."
    }
    $python = $command.Source
    Write-Warning "Окружение .venv не найдено, используется системный Python: $python"
}

$arguments = @("-m", "mira.ui", "--port", "$Port")
if ($Chat) { $arguments += "--chat" }
if ($Typing) { $arguments += "--typing-delay" }
if ($Retrieval) { $arguments += "--retrieval" }
if ($NoIntent) { $arguments += "--no-intent" }
if ($NoTypos) { $arguments += "--no-typos" }
if ($NoBrowser) { $arguments += "--no-browser" }

& $python @arguments
