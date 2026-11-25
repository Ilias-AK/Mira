param(
    [ValidateSet('all','language','dialogue')][string]$Stage = 'all',
    [ValidateSet('small','main','large')][string]$Preset = 'main',
    [string]$RunName = 'main',
    [int]$BatchSize = 4,
    [int]$Accumulation = 8,
    [double]$LanguageHours = 12,
    [double]$DialogueHours = 18,
    [int]$LanguageSteps = 12000,
    [int]$DialogueSteps = 16000,
    [ValidateRange(1,8)][int]$CacheWorkers = 4,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain letters, digits, _ or -.' }
if ($BatchSize -lt 1 -or $Accumulation -lt 1) { throw 'Batch and accumulation must be positive.' }
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Run scripts/bootstrap_lenovo.ps1 first.' }
function Invoke-PythonStep([string[]]$Arguments) {
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training step failed: $($Arguments -join ' ')" }
}
Invoke-PythonStep @('-m','arena.doctor','--require-cuda','--output',"runs/$RunName/environment.json")
foreach ($required in @('data/processed/train.jsonl','data/processed/val.jsonl','artifacts/tokenizer.json','artifacts/retrieval/index.sqlite3')) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Prepared input missing: $required. Run scripts/prepare_data.ps1 first." }
}
$common = @('-m','arena.training','--data','data/processed/train.jsonl','--val-data','data/processed/val.jsonl',
    '--tokenizer','artifacts/tokenizer.json','--preset',$Preset,'--device','cuda',
    '--batch-size',"$BatchSize",'--accumulation',"$Accumulation",'--threads','4',
    '--max-source-length','512','--max-target-length','96',
    '--eval-every','200','--save-every','100','--eval-batches','20','--log-every','20')
$languageDir = "runs/$RunName/language"
$dialogueDir = "runs/$RunName/dialogue"
if ($Stage -eq 'all' -or $Stage -eq 'language') {
    $languageArgs = $common + @('--stage','language','--output-dir',$languageDir,
        '--hours',"$LanguageHours",'--steps',"$LanguageSteps")
    $last = Join-Path $languageDir 'last.pt'
    if (Test-Path -LiteralPath $last) {
        if (-not $Resume) { throw "Existing checkpoint $last. Use -Resume or a new -RunName." }
        $languageArgs += @('--resume',$last)
    }
    Invoke-PythonStep $languageArgs
}
if ($Stage -eq 'all' -or $Stage -eq 'dialogue') {
    $dialogueArgs = $common + @('--stage','dialogue','--output-dir',$dialogueDir,
        '--hours',"$DialogueHours",'--steps',"$DialogueSteps",'--lr','0.00015',
        '--retrieval-index','artifacts/retrieval','--retrieval-cache','artifacts/retrieval_cache.sqlite3',
        '--retrieval-dropout','0.25')
    $last = Join-Path $dialogueDir 'last.pt'
    if (Test-Path -LiteralPath $last) {
        if (-not $Resume) { throw "Existing checkpoint $last. Use -Resume or a new -RunName." }
        $dialogueArgs += @('--resume',$last)
    } else {
        $languageBest = Join-Path $languageDir 'best.pt'
        if (-not (Test-Path -LiteralPath $languageBest)) { throw "Language checkpoint not found: $languageBest" }
        $dialogueArgs += @('--init-from',$languageBest)
    }
    Write-Host 'Checking retrieval cache. A complete matching cache is reused without searching again.'
    Invoke-PythonStep @('-m','arena.retrieval_cache','build','--workers',"$CacheWorkers")
    Invoke-PythonStep $dialogueArgs
}
Write-Host "Training stage finished. Check runs/$RunName/*/metrics.jsonl and best.pt before choosing a release."

