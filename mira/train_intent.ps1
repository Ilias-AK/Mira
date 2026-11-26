# Переобучение нейросети тем по рецепту v2 (данные команды + пакет v2, независимая проверка). 15–20 минут.
# Новая модель пишется в runs\mira_intent_next и сравнивается с рабочей; рабочую НЕ заменяет.
#   powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\train_intent.ps1
param([int]$Epochs = 40, [string]$Checkpoint = "runs\main\dialogue\best.pt")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$root"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m mira.train_intent --checkpoint $Checkpoint --tokenizer artifacts\tokenizer.json --epochs $Epochs --seed 7 `
    --bank config\bank.json --extra-train data\delivery_v2\intent\train.jsonl --val-file data\delivery_v2\intent\val.jsonl `
    --out runs\mira_intent_next\intent.pt
if ($LASTEXITCODE -ne 0) { Write-Host "Обучение не прошло" -ForegroundColor Red; exit 1 }
& $py -m mira.eval_candidate --name current --config-dir config --intent runs\mira_intent\intent.pt
& $py -m mira.eval_candidate --name next --config-dir config --intent runs\mira_intent_next\intent.pt
& $py -m mira.eval_candidate --compare runs\eval_v2\current.json runs\eval_v2\next.json
Write-Host "Если вердикт хороший: скопируйте runs\mira_intent_next\intent.* в runs\mira_intent\" -ForegroundColor Green
