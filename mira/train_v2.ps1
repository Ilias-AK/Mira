# Обучение и проверка кандидата v2 на пакете mira_delivery_clean_v2.1. Рабочую Миру НЕ заменяет.
# Запуск (из любой папки):
#   powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\train_v2.ps1
#   ... -SkipGenerator          без генератора (только нейросеть тем и проверки, ~15 мин)
#   ... -Zip D:\path\mira_delivery_clean_v2.1.zip   заново собрать data\delivery_v2 из архива (архив не меняется)
param([string]$Zip = "", [switch]$SkipGenerator, [int]$IntentEpochs = 40, [int]$GenSteps = 2500, [double]$GenHours = 1.5)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$root"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$logs = Join-Path $root "runs\train_v2\$stamp"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
function Step($title, [string[]]$arguments, $log) {
    Write-Host "`n=== $title" -ForegroundColor Cyan
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # предупреждения в stderr не должны обрывать скрипт
    & $py @arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logs $log)
    $code = $LASTEXITCODE
    $ErrorActionPreference = $saved
    if ($code -ne 0) { Write-Host "Шаг не прошёл: $title (лог: $logs\$log)" -ForegroundColor Red; exit 1 }
}

# 0. baseline: копия действующей модели и конфигов (один раз)
$base = Join-Path $root "runs\baseline_v1"
if (-not (Test-Path (Join-Path $base "intent.pt"))) {
    New-Item -ItemType Directory -Force -Path (Join-Path $base "config") | Out-Null
    Copy-Item runs\mira_intent\intent.pt, runs\mira_intent\intent.json $base
    Copy-Item config\*.json (Join-Path $base "config")
    Write-Host "Baseline сохранён в runs\baseline_v1" -ForegroundColor Green
}

# 1. данные (по умолчанию уже собраны и лежат в data\delivery_v2)
if ($Zip) { Step "Импорт архива" @("-m", "mira.import_delivery", "--zip", $Zip) "01_import.log" }
Get-FileHash data\delivery_v2\manifest.json -Algorithm SHA256 | Format-List | Out-File (Join-Path $logs "manifest_hash.txt")

# 2. метрики baseline (действующая модель тем + рабочий профиль)
Step "Baseline: нейросеть тем" @("-m", "mira.eval_intent_model", "--model", "runs\baseline_v1\intent.pt", "--name", "intent_baseline") "02_intent_baseline.log"
Step "Baseline: Мира целиком" @("-m", "mira.eval_candidate", "--name", "baseline", "--config-dir", "runs\baseline_v1\config", "--intent", "runs\baseline_v1\intent.pt") "03_eval_baseline.log"

# 3. новая нейросеть тем (в отдельную папку)
Step "Обучение нейросети тем v2" @("-m", "mira.train_intent", "--checkpoint", "runs\main\dialogue\best.pt", "--tokenizer", "artifacts\tokenizer.json",
    "--bank", "config\candidate_v2_keep\bank.json", "--extra-train", "data\delivery_v2\intent\train.jsonl",
    "--val-file", "data\delivery_v2\intent\val.jsonl", "--out", "runs\mira_intent_v2\intent.pt", "--epochs", "$IntentEpochs", "--seed", "7") "04_train_intent.log"
Step "Кандидат: нейросеть тем" @("-m", "mira.eval_intent_model", "--model", "runs\mira_intent_v2\intent.pt", "--name", "intent_candidate") "05_intent_candidate.log"
Step "Кандидат (ответы пакета)" @("-m", "mira.eval_candidate", "--name", "candidate_v2", "--config-dir", "config\candidate_v2", "--intent", "runs\mira_intent_v2\intent.pt") "06_eval_candidate_v2.log"
Step "Кандидат (утверждённые ответы сохранены)" @("-m", "mira.eval_candidate", "--name", "candidate_v2_keep", "--config-dir", "config\candidate_v2_keep", "--intent", "runs\mira_intent_v2\intent.pt") "07_eval_candidate_keep.log"
Step "Сравнение: baseline -> candidate_v2" @("-m", "mira.eval_candidate", "--compare", "runs\eval_v2\baseline.json", "runs\eval_v2\candidate_v2.json") "08_compare_v2.log"
Copy-Item runs\eval_v2\comparison.md runs\eval_v2\comparison_candidate_v2.md -Force
Step "Сравнение: baseline -> candidate_v2_keep" @("-m", "mira.eval_candidate", "--compare", "runs\eval_v2\baseline.json", "runs\eval_v2\candidate_v2_keep.json") "09_compare_keep.log"
Copy-Item runs\eval_v2\comparison.md runs\eval_v2\comparison_candidate_v2_keep.md -Force

# 4. генератор: дообучение собственного (с нуля обученного) runs\main\dialogue\best.pt на диалогах Миры
if (-not $SkipGenerator) {
    Step "Обучение генератора v2 (до $GenHours ч)" @("-m", "arena.training", "--stage", "dialogue", "--init-checkpoint", "runs\main\dialogue\best.pt",
        "--data", "data\delivery_v2\generator\train_fit.jsonl", "--val-data", "data\delivery_v2\generator\dev_from_train.jsonl",
        "--tokenizer", "artifacts\tokenizer.json", "--output-dir", "runs\generator_v2", "--steps", "$GenSteps", "--hours", "$GenHours",
        "--batch-size", "4", "--accumulation", "8", "--lr", "0.0001", "--warmup-steps", "100", "--eval-every", "250",
        "--save-every", "250", "--eval-batches", "100", "--seed", "42", "--retrieval-dropout", "0", "--device", "cuda") "10_train_generator.log"
    Step "Генератор baseline на validation" @("-m", "mira.eval_generator", "--checkpoint", "runs\main\dialogue\best.pt", "--name", "gen_baseline") "11_gen_baseline.log"
    Step "Генератор v2 на validation" @("-m", "mira.eval_generator", "--checkpoint", "runs\generator_v2\best.pt", "--name", "gen_candidate") "12_gen_candidate.log"
}

# 5. старые тесты Миры (рабочий профиль не должен сломаться)
Step "Самопроверка Миры" @((Join-Path $here "tests_mira.py")) "13_tests_mira.log"

Copy-Item runs\eval_v2\*.json, runs\eval_v2\*.md $logs -Force
if (Test-Path runs\mira_intent_v2\intent.json) { Copy-Item runs\mira_intent_v2\intent.json (Join-Path $logs "intent_v2_report.json") }
Write-Host "`nГотово. Все логи и метрики: $logs" -ForegroundColor Green
Write-Host "Рабочая Мира не изменена. Пришлите папку $logs (или файлы comparison_*.md и *.json) для разбора." -ForegroundColor Green
