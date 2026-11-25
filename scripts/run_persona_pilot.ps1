param(
    [string]$Checkpoint = 'runs/main/dialogue/best.pt',
    [string]$RunName = 'persona_pilot',
    [ValidateRange(1,1000000)][int]$Steps = 2000,
    [ValidateRange(1,64)][int]$BatchSize = 4,
    [ValidateRange(1,128)][int]$Accumulation = 8,
    [ValidateRange(1,1000000)][int]$ReviewPerSource = 2000,
    [ValidateRange(1,100000)][int]$GenerationExamples = 200,
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda',
    [string]$PythonPath = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Use letters, digits, _ or - for RunName.' }
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($PythonPath) { $venvPython = $PythonPath }
foreach ($file in @($venvPython, $Checkpoint, 'data/processed/train.jsonl', 'data/processed/val.jsonl',
                    'artifacts/tokenizer.json', 'artifacts/retrieval_cache.sqlite3')) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file missing: $file" }
}
$runRoot = "runs/$RunName"
if ((Test-Path -LiteralPath $runRoot) -and -not $Resume) {
    throw 'Run directory already exists. Use -Resume with identical settings or a new -RunName.'
}
$checkpointHash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
$settings = [ordered]@{ checkpoint_sha256=$checkpointHash; steps=$Steps; batch_size=$BatchSize;
    accumulation=$Accumulation; seed=42; lr=0.00005; persona_dropout=0.3;
    persona_shuffle=$true; evaluation_per_source=$ReviewPerSource; generation_examples=$GenerationExamples;
    device=$Device; version=1 }
$settingsPath = Join-Path $runRoot 'pilot_settings.json'
if (Test-Path -LiteralPath $settingsPath) {
    $previous = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
    foreach ($key in $settings.Keys) {
        if ($previous.$key -ne $settings[$key]) { throw "Pilot setting changed: $key. Use a new -RunName." }
    }
} else {
    if ($Resume -and (Test-Path -LiteralPath $runRoot)) { throw 'Existing run lacks pilot_settings.json.' }
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    $settings | ConvertTo-Json | Set-Content -LiteralPath $settingsPath -Encoding UTF8
}
function Invoke-PythonStep([string[]]$Arguments) {
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $($Arguments -join ' ')" }
}
function Review-Checkpoint([string]$ModelPath, [string]$Label) {
    $digest = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $reportPath = Join-Path $runRoot "$Label-$($digest.Substring(0,12)).json"
    if (Test-Path -LiteralPath $reportPath) {
        $saved = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ($saved.inputs.checkpoint_sha256 -ne $digest) { throw 'Review checkpoint identity mismatch.' }
        Write-Host "Reusing existing review: $reportPath"
    } else {
        Invoke-PythonStep @('-m','arena.generation_review','--checkpoint',$ModelPath,'--device',$Device,
            '--max-per-source',"$ReviewPerSource",'--generation-examples',"$GenerationExamples",'--seed','42','--output',$reportPath)
    }
    $conversationPath = Join-Path $runRoot "conversation-$Label-$($digest.Substring(0,12)).json"
    if (-not (Test-Path -LiteralPath $conversationPath)) {
        Invoke-PythonStep @('-m','arena.evaluation','run','--rounds','configs/conversation_review_rounds.json',
            '--checkpoint',$ModelPath,'--device',$Device,'--output',$conversationPath)
    }
}
$doctorArgs = @('-m','arena.doctor','--output',"$runRoot/environment.json")
if ($Device -eq 'cuda') { $doctorArgs += '--require-cuda' }
Invoke-PythonStep $doctorArgs
Review-Checkpoint $Checkpoint 'baseline'
# Both branches start from the identical saved weights, reset optimizer/counters,
# and use the full frozen corpus. This is a paired experiment, not a new release.
$common = @('-m','arena.training','--data','data/processed/train.jsonl','--val-data','data/processed/val.jsonl',
    '--tokenizer','artifacts/tokenizer.json','--stage','dialogue','--device',$Device,
    '--steps',"$Steps",'--batch-size',"$BatchSize",'--accumulation',"$Accumulation",'--threads','4',
    '--seed','42','--lr','0.00005','--warmup-steps','100',
    '--eval-every','500','--eval-batches',"$([int][Math]::Ceiling($ReviewPerSource / [double]$BatchSize))",'--validation-sampling','fixed-shuffle',
    '--save-every','100','--log-every','20','--retrieval-index','artifacts/retrieval',
    '--retrieval-cache','artifacts/retrieval_cache.sqlite3','--retrieval-dropout','0.25')
foreach ($branch in @('control','augmentation')) {
    $branchDir = Join-Path $runRoot $branch
    $branchArgs = $common + @('--output-dir',$branchDir)
    if ($branch -eq 'control') { $branchArgs += @('--persona-dropout','0','--no-persona-shuffle') }
    else { $branchArgs += @('--persona-dropout','0.3','--persona-shuffle') }
    $last = Join-Path $branchDir 'last.pt'
    if (Test-Path -LiteralPath $last) {
        if (-not $Resume) { throw "Existing branch checkpoint: $last" }
        $branchArgs += @('--resume',$last)
    } else { $branchArgs += @('--init-from',$Checkpoint) }
    Invoke-PythonStep $branchArgs
    Review-Checkpoint (Join-Path $branchDir 'best.pt') $branch
}
Write-Host "Pilot finished: $runRoot. Compare actual answers in all three reviews before choosing any checkpoint."
