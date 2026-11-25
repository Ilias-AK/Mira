@echo off
chcp 65001 >nul
title Mira
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0mira\run_mira.ps1" %*
pause
