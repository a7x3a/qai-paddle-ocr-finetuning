<#
.SYNOPSIS
    Train the Kurdish OCR recognition model for one pipeline phase.

.DESCRIPTION
    Wraps PaddleOCR's tools/train.py with the overrides this project needs, picks batch
    size and dataloader workers from the actual hardware, keeps a separate output
    directory per phase so smoke/pilot/full never overwrite each other, and prints a
    throughput/memory summary when the run finishes.

    Phases (defaults, override with -Epochs/-Limit):
      smoke  1 epoch over a 2,000-sample slice in FP32. Proves the whole loop works.
      pilot  8 epochs over the full training set with AMP. Measures real throughput and
             gives an early read on whether fine-tuning beats the base model.
      full   40 epochs over the full training set with AMP.

.EXAMPLE
    .\scripts\train.ps1 -Phase smoke
.EXAMPLE
    .\scripts\train.ps1 -Phase pilot
.EXAMPLE
    .\scripts\train.ps1 -Phase full -Epochs 60 -BatchSize 48
#>
[CmdletBinding()]
param(
    [ValidateSet('smoke', 'pilot', 'full')]
    [string]$Phase = 'full',

    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$Workers = 0,
    [int]$PrintBatchStep = 20,

    # Train on only the first N training samples. Used by smoke to stay fast.
    [int]$Limit = 0,

    # FP32 instead of mixed precision. The smoke phase runs FP32 on purpose so a
    # mixed-precision numerical problem can never hide behind the fast path.
    [switch]$NoAmp,

    [switch]$Resume,
    [switch]$Cpu,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$projectRoot = Get-QaiProjectRoot
$paddleRoot = Get-QaiPaddleRoot
$python = Get-QaiPython
$config = Join-Path $projectRoot 'configs\kurdish_rec.yml'
$dataset = Join-Path $projectRoot 'data\kurdish_rec'
$pretrained = Join-Path $projectRoot 'pretrain_models\arabic_PP-OCRv5_mobile_rec_pretrained.pdparams'
$outputDir = Join-Path $projectRoot "output\kurdish_$Phase"

# ---------------------------------------------------------------- phase defaults
switch ($Phase) {
    'smoke' {
        if ($Epochs -le 0) { $Epochs = 1 }
        # Enough samples for ~35 optimiser steps at the measured batch of 384. A smaller
        # slice produced only ~5 steps, which cannot show a trend.
        if ($Limit -le 0) { $Limit = 12000 }
    }
    'pilot' {
        if ($Epochs -le 0) { $Epochs = 8 }
    }
    'full' {
        if ($Epochs -le 0) { $Epochs = 40 }
    }
}

# ---------------------------------------------------------------- hardware
$hardware = Get-QaiHardwareProfile
if ($BatchSize -le 0) { $BatchSize = $hardware.RecommendedBatch }
if ($Workers -le 0) { $Workers = $hardware.RecommendedWorkers }
# MultiScaleSampler divides first_bs by every divided_factor entry, so it must be a
# multiple of 16. Silently correcting beats letting the sampler crash.
$BatchSize = [math]::Max(16, [int]([math]::Floor($BatchSize / 16) * 16))

# Validation must NOT use the training batch size. eval_batch_step runs the model
# in eval mode while the training allocator still holds its pool, so a 384-wide eval
# batch on top of a 4.4 GB reserved training pool is what exhausted a 6 GB card at
# step 1000. Evaluation has no gradients, so a small batch costs almost nothing:
# 13,981 val images at 96 measured ~350 img/s, about 40 s.
if ($EvalBatchSize -le 0) { $EvalBatchSize = [math]::Min(96, $BatchSize) }
$ampLevel = if ($NoAmp) { 'O0' } else { 'O2' }

# ---------------------------------------------------------------- preflight
foreach ($path in @($paddleRoot, $python, $config, $dataset, $pretrained)) {
    if (-not (Test-Path $path)) { throw "Required path is missing: $path  (run scripts\setup.ps1)" }
}
if (-not $SkipVerify) {
    $verifyArgs = @{}
    if (-not $Cpu) { $verifyArgs.RequireCuda = $true }
    [void](Assert-QaiEnvironment @verifyArgs)
}

# A phase-specific subset keeps the smoke phase honest (it still reads real images from
# disk through the real pipeline) without paying for 117k samples.
$trainList = 'train_rec.txt'
if ($Limit -gt 0) {
    $subset = Join-Path $dataset "train_subset_$($Limit).txt"
    if (-not (Test-Path $subset)) {
        $lines = [System.IO.File]::ReadAllLines((Join-Path $dataset 'train_rec.txt'))
        $take = [math]::Min($Limit, $lines.Count)
        [System.IO.File]::WriteAllLines($subset, $lines[0..($take - 1)], (New-Object System.Text.UTF8Encoding $false))
    }
    $trainList = "train_subset_$($Limit).txt"
}

# A stale training or probe process still holding VRAM makes the next run die with an
# opaque "available memory is only 0.000000B" within seconds of starting. Wait it out.
if (-not $Cpu) {
    $smi = Get-QaiNvidiaSmi
    if ($smi) {
        $used = [int](& $smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        if ($used -gt 1024) {
            Write-Host "  waiting for the GPU to free up (${used} MB in use)..." -ForegroundColor Yellow
            for ($wait = 0; $wait -lt 120; $wait++) {
                Start-Sleep -Seconds 5
                $used = [int](& $smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
                if ($used -le 1024) { break }
            }
            if ($used -gt 1024) {
                throw "GPU still holds ${used} MB after 10 minutes. Close any running training, probe or evaluation process and retry."
            }
            Write-QaiOk 'GPU is free'
        }
    }
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Write-QaiHeader "Kurdish OCR training | phase '$Phase'"
Write-Host "  gpu        $($hardware.GpuName) ($($hardware.GpuMemoryMB) MB)" -ForegroundColor DarkGray
Write-Host "  cpu / ram  $($hardware.CpuCores) cores / $($hardware.RamGB) GB" -ForegroundColor DarkGray
Write-Host "  batch      $BatchSize   eval batch $EvalBatchSize   workers $Workers   amp $ampLevel   epochs $Epochs" -ForegroundColor DarkGray
Write-Host "  output     $outputDir" -ForegroundColor DarkGray
if ($Limit -gt 0) { Write-Host "  subset     $trainList ($Limit samples)" -ForegroundColor DarkGray }
Write-Host ''

# ---------------------------------------------------------------- overrides
# All paths are relative to the PaddleOCR checkout, which is the working directory below.
$overrides = @(
    "Global.save_model_dir=../output/kurdish_$Phase",
    'Global.pretrained_model=../pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained',
    "Global.epoch_num=$Epochs",
    "Global.use_gpu=$(-not $Cpu)",
    "Global.amp_level=$ampLevel",
    "Global.print_batch_step=$PrintBatchStep",
    "Global.save_epoch_step=1",
    "Train.sampler.first_bs=$BatchSize",
    "Train.loader.batch_size_per_card=$BatchSize",
    "Train.loader.num_workers=$Workers",
    "Eval.loader.batch_size_per_card=$EvalBatchSize",
    "Eval.loader.num_workers=$Workers",
    "Train.dataset.label_file_list=['../data/kurdish_rec/$trainList']"
)

if ($Resume) {
    $latest = Join-Path $outputDir 'latest.pdparams'
    $best = Join-Path $outputDir 'best_accuracy.pdparams'
    if (Test-Path $latest) {
        $overrides += "Global.checkpoints=../output/kurdish_$Phase/latest"
        Write-Host "  resuming from latest" -ForegroundColor Yellow
    }
    elseif (Test-Path $best) {
        $overrides += "Global.checkpoints=../output/kurdish_$Phase/best_accuracy"
        Write-Host "  resuming from best_accuracy" -ForegroundColor Yellow
    }
    else {
        throw "Cannot resume: no latest.pdparams or best_accuracy.pdparams in $outputDir"
    }
}

# ---------------------------------------------------------------- run
$started = Get-Date
Push-Location $paddleRoot
# Paddle and its CUDA runtime write progress notes to stderr. Under
# ErrorActionPreference=Stop PowerShell turns those into terminating errors even when
# training succeeds, so relax it around the native call and trust the exit code.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $python tools\train.py -c $config @('-o') $overrides
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousPreference
    Pop-Location
}
$elapsed = (Get-Date) - $started

if ($exitCode -ne 0) {
    Write-QaiFail "training exited with code $exitCode after $([math]::Round($elapsed.TotalMinutes, 1)) min"
    Write-Host "  log: $(Join-Path $outputDir 'train.log')" -ForegroundColor DarkGray
    exit $exitCode
}

# ---------------------------------------------------------------- summary
$logPath = Join-Path $outputDir 'train.log'
Write-Host ''
Write-QaiOk ("training finished in {0:N1} min" -f $elapsed.TotalMinutes)
Write-Host "  log: $logPath" -ForegroundColor DarkGray

& $python (Join-Path $PSScriptRoot 'benchmark.py') --phase $Phase --train-only
exit 0
