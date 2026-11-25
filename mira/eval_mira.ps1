# Проверка Миры: самотест + прогон тестовых раундов. Результаты в runs\mira\
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$here;$root"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py (Join-Path $here "tests_mira.py")
if ($LASTEXITCODE -ne 0) { Write-Host "Самотест не прошёл" -ForegroundColor Red; exit 1 }
$sets = @(
  @{name="dev";    file=(Join-Path $root "night_v4\configs\night_v4_dev_rounds.json")},
  @{name="closed"; file=(Join-Path $root "night_v4\configs\night_v4_closed_rounds.json")},
  @{name="blind";  file=(Join-Path $here "config\blind_rounds.json")},
  @{name="hall";   file=(Join-Path $here "config\hall_blind.json")}
)
foreach ($s in $sets) {
  if (Test-Path $s.file) {
    Write-Host "== $($s.name)" -ForegroundColor Cyan
    & $py -m mira.eval_dev --rounds $s.file --out (Join-Path $root "runs\mira\$($s.name)")
  }
}
Write-Host "== проверочные вопросы команды" -ForegroundColor Cyan
& $py -m mira.eval_questions
Write-Host "Готово. Транскрипты: runs\mira\*\transcript.md, отчёт по вопросам: runs\mira\questions\report.md" -ForegroundColor Green
