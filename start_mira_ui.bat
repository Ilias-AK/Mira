@echo off
chcp 65001 >nul
title Mira 1.0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0mira\run_ui.ps1" %*
pause
