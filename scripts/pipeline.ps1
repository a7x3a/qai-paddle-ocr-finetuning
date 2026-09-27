<#
.SYNOPSIS
    Run the whole smoke -> pilot -> full pipeline end to end.

.DESCRIPTION
    Each phase trains into its own directory, exports to its own model directory and is
    scored on the validation split, then compared against the base model. The full phase
    is only started when the pilot actually beat the baseline, so a bad hyper-parameter
    change is caught in a couple of hours instead of a day.

    -From and -To bound which phases run, so a run can be resumed after a failure:
      .\scripts\pipeline.ps1 -From full

.EXAMPLE
    .\scripts\pipeline.ps1
.EXAMPLE
    .\scripts\pipeline.ps1 -From pilot -To pilot
#>
[CmdletBinding()]
param(
    [ValidateSet('smoke', 'pilot', 'full')]
    [string]$From = 'smoke',

    [ValidateSet('smoke', 'pilot', 'full')]
    [string]$To = 'full',

    [int]$FullEpochs = 0,
    [int]$BatchSize = 0,
    [switch]$SkipBaseEvaluation,
    [switch]$ContinueOnPilotFailure
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$projectRoot = Get-QaiProjectRoot
$python = Get-QaiPython
$order = @('smoke', 'pilot', 'full')
$startIndex = $order.IndexOf($From)
$endIndex = $order.IndexOf($To)
if ($startIndex -gt $endIndex) { throw "-From ($From) comes after -To ($To)" }

$baselineReport = Join-Path $projectRoot 'reports\val_base_arabic_ppocrv5.json'

function Get-PhaseCER {
    param([string]$Phase)
    $path = Join-Path $projectRoot "reports\val_kurdish_$Phase.json"
    if (-not (Test-Path $path)) { return $null }
    return ((Get-Content $path -Raw | ConvertFrom-Json).overall.cer)
}

# ---------------------------------------------------------------- base baseline
Write-QaiHeader 'Phase 0 | base model baseline'
if ($SkipBaseEvaluation -and (Test-Path $baselineReport)) {
    Write-QaiStep "using existing $baselineReport"
}
elseif (Test-Path $baselineReport) {
    Write-QaiStep "using existing $baselineReport"
}
else {
    & $python (Join-Path $PSScriptRoot 'evaluate.py') `
        --model-dir (Join-Path $projectRoot 'assets\base_rec_inference') `
        --split val --name base_arabic_ppocrv5
    if ($LASTEXITCODE -ne 0) { throw 'base model evaluation failed' }
}
$baseCER = Get-PhaseCER -Phase 'base'
Write-QaiOk "base model val CER: $([math]::Round($baseCER * 100, 2))%"

# ---------------------------------------------------------------- phases
for ($index = $startIndex; $index -le $endIndex; $index++) {
    $phase = $order[$index]

    Write-QaiHeader "Phase $($index + 1) | $phase"
    $trainArgs = @('-Phase', $phase)
    if ($phase -eq 'full' -and $FullEpochs -gt 0) { $trainArgs += @('-Epochs', $FullEpochs) }
    if ($BatchSize -gt 0) { $trainArgs += @('-BatchSize', $BatchSize) }

    & (Join-Path $PSScriptRoot 'train.ps1') @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "training failed in phase $phase" }

    & (Join-Path $PSScriptRoot 'export_model.ps1') -Phase $phase
    if ($LASTEXITCODE -ne 0) { throw "export failed in phase $phase" }

    & $python (Join-Path $PSScriptRoot 'evaluate.py') `
        --model-dir (Join-Path $projectRoot "export\kurdish_$phase") `
        --split val --name "kurdish_$phase" --no-csv `
        --compare-to $baselineReport
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed in phase $phase" }

    $cer = Get-PhaseCER -Phase $phase
    Write-QaiOk ("{0} val CER: {1:N4}%  (base {2:N4}%, {3:+0.00;-0.00} pp)" -f `
        $phase, ($cer * 100), ($baseCER * 100), (($cer - $baseCER) * 100))

    & $python (Join-Path $PSScriptRoot 'benchmark.py') --phase $phase --compare-to $baselineReport
    if ($LASTEXITCODE -ne 0) { Write-QaiStep 'benchmark report failed (non-fatal)' }

    # The pilot is the go/no-go gate for the long run.
    if ($phase -eq 'pilot' -and $cer -ge $baseCER -and -not $ContinueOnPilotFailure) {
        Write-Host ''
        Write-QaiFail ("pilot did not beat the base model ({0:N4}% vs {1:N4}%)." -f ($cer * 100), ($baseCER * 100))
        Write-Host '  Check reports\val_kurdish_pilot.md for the per-domain breakdown, then' -ForegroundColor DarkGray
        Write-Host '  adjust the learning rate or freeze epochs in configs\kurdish_rec.yml.' -ForegroundColor DarkGray
        Write-Host '  Re-run with -ContinueOnPilotFailure to override.' -ForegroundColor DarkGray
        exit 1
    }
}

# ---------------------------------------------------------------- summary
Write-QaiHeader 'Pipeline complete'
& $python (Join-Path $PSScriptRoot 'benchmark.py') --summary
Write-Host ''
Write-Host '  reports\' -ForegroundColor Cyan
Get-ChildItem (Join-Path $projectRoot 'reports') -Filter '*.md' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 8 |
    ForEach-Object { Write-Host "    $($_.Name)" -ForegroundColor DarkGray }
exit 0
