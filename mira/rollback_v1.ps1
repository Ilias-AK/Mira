# Откат рабочей Миры к версии до v2 (копия в runs\baseline_v1, сделана train_v2.ps1 26.11.2026).
#   powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\rollback_v1.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
if (-not (Test-Path runs\baseline_v1\intent.pt)) { Write-Host "Нет runs\baseline_v1 — откатывать не из чего" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path runs\v2_backup\config | Out-Null
Copy-Item config\*.json runs\v2_backup\config -Force
Copy-Item runs\mira_intent\intent.pt, runs\mira_intent\intent.json runs\v2_backup -Force
Copy-Item runs\baseline_v1\config\*.json config -Force
Copy-Item runs\baseline_v1\intent.pt, runs\baseline_v1\intent.json runs\mira_intent -Force
Write-Host "Готово: рабочая Мира снова v1. Версия v2 сохранена в runs\v2_backup." -ForegroundColor Green
