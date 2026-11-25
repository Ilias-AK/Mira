param(
    [string]$WikiShards = '1',
    [long]$MaxWikiChars = 200000000,
    [switch]$WithoutToloka,
    [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Run scripts/bootstrap_lenovo.ps1 first.' }
function Invoke-PythonStep([string[]]$Arguments) {
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline step failed: $($Arguments -join ' ')" }
}
$sources = if ($WithoutToloka) { @('oasst','wikipedia') } else { @('toloka','oasst','wikipedia') }
if (-not $SkipDownload) {
    Invoke-PythonStep (@('-m','arena.data','fetch','--wiki-shards',$WikiShards,'--sources') + $sources)
}
$prepareArgs = @('-m','arena.data','prepare','--wiki-shards',$WikiShards,'--max-wiki-chars',"$MaxWikiChars",'--sources') + $sources
if (-not $WithoutToloka) { $prepareArgs += '--include-toloka' }
Invoke-PythonStep $prepareArgs
Invoke-PythonStep @('-m','arena.data','validate','--output-dir','data/processed')
Invoke-PythonStep @('-m','arena.tokenization','train','--input','data/processed/train.jsonl','--output','artifacts/tokenizer.json','--vocab-size','8192')
Invoke-PythonStep @('-m','arena.retrieval','build','--data','data/processed/train.jsonl','--output','artifacts/retrieval')
Invoke-PythonStep @('-m','arena.retrieval','fit-reranker','--index','artifacts/retrieval','--auto-negatives','--max-examples','4000')
Write-Host 'Data, scratch tokenizer, retrieval index and initial reranker are ready.'
Write-Host 'A changed tokenizer requires a new generator run; checkpoint hashes enforce this.'

