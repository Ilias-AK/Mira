# Только проверка (без обучения): baseline и кандидаты с уже обученной нейросетью тем v2. ~8 минут.
#   powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\eval_v2.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$root"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root ("runs\train_v2\eval-" + (Get-Date -Format "yyyyMMdd-HHmm"))
New-Item -ItemType Directory -Force -Path $logs | Out-Null
function Step($title, [string[]]$arguments, $log) {
    Write-Host "`n=== $title" -ForegroundColor Cyan
    $saved = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $py @arguments 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath (Join-Path $logs $log) -Encoding utf8
    $code = $LASTEXITCODE; $ErrorActionPreference = $saved
    Get-Content (Join-Path $logs $log) -Encoding utf8 | Select-Object -Last 4
    if ($code -ne 0) { Write-Host "Шаг не прошёл: $title" -ForegroundColor Red; exit 1 }
}
Step "Baseline" @("-m", "mira.eval_candidate", "--name", "baseline", "--config-dir", "runs\baseline_v1\config", "--intent", "runs\baseline_v1\intent.pt") "01_baseline.log"
Step "Кандидат v2 (ответы пакета)" @("-m", "mira.eval_candidate", "--name", "candidate_v2", "--config-dir", "config\candidate_v2", "--intent", "runs\mira_intent_v2\intent.pt") "02_candidate_v2.log"
Step "Кандидат v2_keep" @("-m", "mira.eval_candidate", "--name", "candidate_v2_keep", "--config-dir", "config\candidate_v2_keep", "--intent", "runs\mira_intent_v2\intent.pt") "03_candidate_keep.log"
Step "Сравнение baseline -> v2" @("-m", "mira.eval_candidate", "--compare", "runs\eval_v2\baseline.json", "runs\eval_v2\candidate_v2.json") "04_compare_v2.log"
Copy-Item runs\eval_v2\comparison.md runs\eval_v2\comparison_candidate_v2.md -Force
Step "Сравнение baseline -> v2_keep" @("-m", "mira.eval_candidate", "--compare", "runs\eval_v2\baseline.json", "runs\eval_v2\candidate_v2_keep.json") "05_compare_keep.log"
Copy-Item runs\eval_v2\comparison.md runs\eval_v2\comparison_candidate_v2_keep.md -Force
Step "Самопроверка" @((Join-Path $here "tests_mira.py")) "06_tests_mira.log"
Copy-Item runs\eval_v2\*.json, runs\eval_v2\*.md $logs -Force
Write-Host "`nГотово: $logs" -ForegroundColor Green
