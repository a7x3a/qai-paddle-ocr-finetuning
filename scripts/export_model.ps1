<#
.SYNOPSIS
    Export a trained phase's checkpoint to a PaddleOCR inference model.

.DESCRIPTION
    Wraps PaddleOCR's tools/export_model.py and always prefers best_accuracy.pdparams over
    latest.pdparams, because the base model is a poor recogniser for this data and a late
    training checkpoint is not automatically the best one.

    The exported directory also receives a copy of the dictionary. Its filename must keep
    the "arabic" substring: PaddleOCR's CTCLabelDecode only reverses Arabic script
    right-to-left when the dictionary path contains it, so a renamed file silently breaks
    every prediction.

.EXAMPLE
    .\scripts\export_model.ps1 -Phase pilot
    .\scripts\export_model.ps1 -Phase full -Checkpoint latest
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('smoke', 'pilot', 'full')]
    [string]$Phase,

    [ValidateSet('best', 'latest')]
    [string]$Checkpoint = 'best'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$projectRoot = Get-QaiProjectRoot
$paddleRoot = Get-QaiPaddleRoot
$python = Get-QaiPython
$config = Join-Path $projectRoot 'configs\kurdish_rec.yml'
$outputDir = Join-Path $projectRoot "output\kurdish_$Phase"
$exportDir = Join-Path $projectRoot "export\kurdish_$Phase"
$ckptName = if ($Checkpoint -eq 'best') { 'best_accuracy' } else { 'latest' }
$source = Join-Path $outputDir "$ckptName.pdparams"

if (-not (Test-Path $source)) {
    $fallback = Join-Path $outputDir 'latest.pdparams'
    if ($Checkpoint -eq 'best' -and (Test-Path $fallback)) {
        Write-Host "  no best_accuracy.pdparams; falling back to latest" -ForegroundColor Yellow
        $ckptName = 'latest'
        $source = $fallback
    }
    else {
        throw "Checkpoint not found: $source  (run scripts\train.ps1 -Phase $Phase first)"
    }
}

New-Item -ItemType Directory -Force -Path $exportDir | Out-Null

Write-QaiHeader "Exporting kurdish_$Phase ($Checkpoint)"
Write-Host "  source $source" -ForegroundColor DarkGray
Write-Host "  target $exportDir" -ForegroundColor DarkGray

$relative = "output/kurdish_$Phase/$ckptName"
$overrides = @(
    "Global.save_inference_dir=../export/kurdish_$Phase",
    "Global.checkpoints=../$relative",
    # The exported yml must carry the same dictionary and max length the model was
    # trained with, or evaluate.py and any predictor will disagree with training.
    'Global.infer_mode=false',
    'Global.use_space_char=true'
)

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
Push-Location $paddleRoot
try {
    & $python tools\export_model.py -c $config @('-o') $overrides
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    $ErrorActionPreference = $previousPreference
}

if ($exitCode -ne 0) {
    throw "export_model.py exited with code $exitCode"
}

$parameters = Join-Path $exportDir 'inference.pdiparams'
if (-not (Test-Path $parameters)) {
    throw "export finished but $parameters is missing"
}

# Keep the RTL-critical filename next to the model so the folder is self-contained.
Copy-Item (Join-Path $projectRoot 'configs\arabic_kurdish_dict.txt') $exportDir -Force

Write-QaiOk ("exported {0:N1} MB" -f ((Get-Item $parameters).Length / 1MB))
Get-ChildItem $exportDir | ForEach-Object { Write-Host "     $($_.Name)" -ForegroundColor DarkGray }
exit 0
