# Чат с Мирой (режим конкурса по умолчанию). Запуск:  powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\run_mira.ps1
# -Typing  пауза как у человека (6 с + 5 символов/с, не больше 35 с)
# -Chat    стиль мессенджера вместо стиля конкурса
# -ShowTime  показывать время формирования ответа (для видео)
# -NoTypos  без редких опечаток (по умолчанию примерно 1 ответ из 25 с одной опечаткой)
# -Retrieval  добавить поиск по базе диалогов (по умолчанию выключен: часто отвечает не по теме)
param([switch]$Retrieval, [switch]$Json, [switch]$Typing, [switch]$Chat, [switch]$NoRetrieval, [switch]$ShowTime, [switch]$NoTypos)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
Set-Location $root
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$here;$root"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$a = @("-m", "mira.chat")
if ($Retrieval) { $a += "--retrieval" }
if ($Json) { $a += "--json" }
if ($Typing) { $a += "--typing-delay" }
if ($Chat) { $a += "--chat" }
if ($ShowTime) { $a += "--show-time" }
if ($NoTypos) { $a += "--no-typos" }
& $py @a
