<#
.SYNOPSIS
    Prepare a Windows machine to train and evaluate the Kurdish OCR model.

.DESCRIPTION
    Idempotent. Safe to re-run: each step is skipped when it is already correct, so this
    doubles as a repair tool on a machine that was set up by an older version.

    Steps:
      1. Create .venv (Python 3.9-3.12)
      2. Install GPU PaddlePaddle, then the data and training requirements
      3. Clone PaddleOCR and check out the exact pinned commit
      4. Verify the committed base models against their checksums
      5. Download the dataset from Hugging Face
      6. Prepare train/val/test label files
      7. Audit the prepared data and verify the environment

    Use -SkipDataset on a machine that will only run inference, and -SkipPaddleInstall if
    Paddle is already present and known-good.

.EXAMPLE
    .\scripts\setup.ps1
.EXAMPLE
    .\scripts\setup.ps1 -SkipDataset -SkipPaddleInstall
#>
[CmdletBinding()]
param(
    [switch]$SkipVenv,
    [switch]$SkipPaddleInstall,
    [switch]$SkipPaddleOcr,
    [switch]$SkipDataset
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$projectRoot = Get-QaiProjectRoot
$venv = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$paddleRoot = Get-QaiPaddleRoot
$commit = '2661c7c0ef5c613e8f93c6e93b2e052399f0f854'
$repo = 'https://github.com/PaddlePaddle/PaddleOCR.git'

Write-QaiHeader 'Kurdish OCR setup'

# ---------------------------------------------------------------- 1. virtualenv
if ($SkipVenv) {
    Write-QaiStep 'skipping virtualenv creation (-SkipVenv)'
}
elseif (Test-Path $venvPython) {
    Write-QaiOk 'virtualenv already present'
}
else {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py -3.12 -m venv $venv }
    else { & python -m venv $venv }
    if (-not (Test-Path $venvPython)) { throw "venv creation failed; expected $venvPython" }
    Write-QaiOk 'created .venv'
}

if (-not (Test-Path $venvPython)) { throw "no interpreter at $venvPython" }
$python = $venvPython

# Paddle and pip write notes to stderr during normal operation.
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'

try {
    & $python -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { throw 'failed to upgrade pip' }
    Write-QaiOk 'pip ready'

    # ------------------------------------------------------------ 2. requirements
    if ($SkipPaddleInstall) {
        Write-QaiStep 'skipping Paddle install (-SkipPaddleInstall)'
    }
    else {
        # PaddlePaddle GPU wheels are published on Paddle's own index, keyed by CUDA
        # version, so it must be installed before the ordinary requirements.
        Write-QaiStep 'installing paddlepaddle-gpu 3.3.1 (CUDA 12.6 wheel)...'
        & $python -m pip install "paddlepaddle-gpu==3.3.1" `
            --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --quiet
        if ($LASTEXITCODE -ne 0) { throw 'failed to install paddlepaddle-gpu' }
        Write-QaiOk 'paddlepaddle-gpu installed'
    }

    foreach ($file in @('requirements-data.txt', 'requirements-training.txt')) {
        Write-QaiStep "installing $file ..."
        & $python -m pip install -r (Join-Path $projectRoot $file) --quiet
        if ($LASTEXITCODE -ne 0) { throw "failed to install $file" }
        Write-QaiOk "$file installed"
    }
}
finally {
    $ErrorActionPreference = $previousPreference
}

# ---------------------------------------------------------------- 3. PaddleOCR
if ($SkipPaddleOcr) {
    Write-QaiStep 'skipping PaddleOCR checkout (-SkipPaddleOcr)'
}
elseif (Test-Path (Join-Path $paddleRoot '.git')) {
    $current = (& git -C $paddleRoot rev-parse HEAD 2>$null)
    if ($current -eq $commit) {
        Write-QaiOk "PaddleOCR already at $commit"
    }
    else {
        Write-QaiStep "PaddleOCR is at $current; checking out pinned $commit"
        & git -C $paddleRoot fetch --all --tags
        & git -C $paddleRoot checkout $commit
        if ($LASTEXITCODE -ne 0) { throw "failed to check out $commit" }
        Write-QaiOk "PaddleOCR pinned to $commit"
    }
}
else {
    Write-QaiStep "cloning PaddleOCR (this takes a few minutes)..."
    & git clone $repo $paddleRoot
    if ($LASTEXITCODE -ne 0) { throw "failed to clone $repo" }
    & git -C $paddleRoot checkout $commit
    if ($LASTEXITCODE -ne 0) { throw "failed to check out $commit" }
    Write-QaiOk "PaddleOCR pinned to $commit"
}

# ---------------------------------------------------------------- 4. base models
Write-QaiStep 'verifying base models...'
& $python (Join-Path $PSScriptRoot 'download_base_models.py')
if ($LASTEXITCODE -ne 0) { throw 'base models check/download failed' }
Write-QaiOk 'base models ready'

# ---------------------------------------------------------------- 5-7. data
if ($SkipDataset) {
    Write-QaiStep 'skipping dataset download and preparation (-SkipDataset)'
}
else {
    $prepared = Join-Path $projectRoot 'data\kurdish_rec\train_rec.txt'
    if (Test-Path $prepared) {
        Write-QaiOk 'prepared dataset already present'
    }
    else {
        Write-QaiStep 'downloading the dataset from Hugging Face (about 850 MB)...'
        & $python (Join-Path $PSScriptRoot 'prepare_dataset.py')
        if ($LASTEXITCODE -ne 0) { throw 'prepare_dataset.py failed' }
        Write-QaiOk 'dataset prepared'
    }

    Write-QaiStep 'auditing the prepared dataset...'
    & $python (Join-Path $PSScriptRoot 'audit_dataset.py')
    if ($LASTEXITCODE -ne 0) { throw 'audit_dataset.py failed; see the report above' }
    Write-QaiOk 'dataset audit passed'
}

# ---------------------------------------------------------------- verify
Write-QaiHeader 'Verifying environment'
& $python (Join-Path $PSScriptRoot 'verify_env.py') --require-data
if ($LASTEXITCODE -ne 0) { throw 'verification failed; fix the reported problems and re-run setup.ps1' }

$hardware = Get-QaiHardwareProfile
Write-Host ''
Write-QaiHeader 'Ready'
Write-Host "  gpu        $($hardware.GpuName) ($($hardware.GpuMemoryMB) MB)" -ForegroundColor DarkGray
Write-Host "  cpu / ram  $($hardware.CpuCores) cores / $($hardware.RamGB) GB" -ForegroundColor DarkGray
Write-Host "  batch      $($hardware.RecommendedBatch)   workers $($hardware.RecommendedWorkers)" -ForegroundColor DarkGray
Write-Host ''
Write-Host '  next steps:' -ForegroundColor Cyan
Write-Host '    .\scripts\train.ps1 -Phase smoke      # fast correctness check' -ForegroundColor DarkGray
Write-Host '    .\scripts\train.ps1 -Phase pilot      # full data, short run' -ForegroundColor DarkGray
Write-Host '    .\scripts\train.ps1 -Phase full       # the real run' -ForegroundColor DarkGray
exit 0
