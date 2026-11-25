# Обучение нейросети тем Миры на вопросах команды (видеокарта, 1–5 минут).
# Данные: data\team_raw\rounds_*.txt и data\team_raw\paraphrases_*.txt
# Запуск:  powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\train_intent.ps1
param([int]$Epochs = 40, [string]$Checkpoint = "runs\main\dialogue\best.pt")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$here;$root"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m mira.team_qa
& $py -m mira.train_intent --checkpoint $Checkpoint --tokenizer artifacts\tokenizer.json --epochs $Epochs
if ($LASTEXITCODE -ne 0) { Write-Host "Обучение не прошло" -ForegroundColor Red; exit 1 }
& $py (Join-Path $here "tests_mira.py")
Write-Host "Готово. Модель: runs\mira_intent\intent.pt. Запускайте run_mira.ps1 — она подключится сама." -ForegroundColor Green
