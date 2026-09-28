<#
.SYNOPSIS
    Prepare a Windows machine to train and evaluate the Kurdish OCR model.

.DESCRIPTION
    Idempotent and resilient: safe to re-run anytime. Automatically detects GPU/CPU,
    installs matching PaddlePaddle wheels, clones PaddleOCR at the pinned commit,
    downloads official pretrained base models, downloads dataset shards from Hugging Face,
    extracts train/val/test splits, audits data, and verifies the full pipeline.

    Steps:
      1. Create or activate .venv (Python 3.9-3.12)
      2. Detect GPU / CUDA and install optimal PaddlePaddle wheel
      3. Clone PaddleOCR repository at the pinned commit
      4. Install unified project dependencies (requirements.txt)
      5. Download and verify official base models (PP-OCRv5)
      6. Download and prepare Kurdish dataset from Hugging Face
      7. Run deep environment and hardware verification

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
    [switch]$SkipDataset,
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$projectRoot = Get-QaiProjectRoot
$venv = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$paddleRoot = Get-QaiPaddleRoot
$commit = '2661c7c0ef5c613e8f93c6e93b2e052399f0f854'
$repo = 'https://github.com/PaddlePaddle/PaddleOCR.git'

Write-QaiHeader 'Kurdish OCR Environment Setup'

# ---------------------------------------------------------------- 1. virtualenv
if ($SkipVenv) {
    Write-QaiStep 'skipping virtualenv creation (-SkipVenv)'
}
elseif (Test-Path $venvPython) {
    Write-QaiOk 'virtualenv already present'
}
else {
    Write-QaiStep 'creating virtual environment .venv...'
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $pyTarget = & py --list 2>$null | Select-String -Pattern "3\.(12|11|10)" | Select-Object -First 1
        if ($pyTarget -match "-(\d+\.\d+)") {
            & py -$($Matches[1]) -m venv $venv
        } else {
            & py -3 -m venv $venv
        }
    }
    else {
        & python -m venv $venv
    }
    if (-not (Test-Path $venvPython)) { throw "venv creation failed; expected $venvPython" }
    Write-QaiOk 'created .venv'
}

if (-not (Test-Path $venvPython)) { throw "no interpreter at $venvPython" }
$python = $venvPython

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'

try {
    Write-QaiStep 'upgrading pip and setuptools...'
    & $python -m pip install --upgrade pip setuptools wheel --quiet
    if ($LASTEXITCODE -ne 0) { throw 'failed to upgrade pip' }
    Write-QaiOk 'pip ready'

    # ------------------------------------------------------------ 2. PaddlePaddle Installation
    if ($SkipPaddleInstall) {
        Write-QaiStep 'skipping Paddle install (-SkipPaddleInstall)'
    }
    else {
        $smi = Get-QaiNvidiaSmi
        $cudaVer = 0
        if ($smi -and (-not $CpuOnly)) {
            $smiOut = & $smi 2>$null | Out-String
            if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
                $cudaVer = [float]"$($Matches[1]).$($Matches[2])"
            }
        }

        if ($cudaVer -ge 12.0 -and (-not $CpuOnly)) {
            Write-QaiStep "detected NVIDIA GPU with CUDA $cudaVer -> installing paddlepaddle-gpu 3.3.1 (cu126)..."
            & $python -m pip install "paddlepaddle-gpu==3.3.1" `
                --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --quiet
        }
        elseif ($cudaVer -ge 11.0 -and (-not $CpuOnly)) {
            Write-QaiStep "detected NVIDIA GPU with CUDA $cudaVer -> installing paddlepaddle-gpu 3.3.1 (cu118)..."
            & $python -m pip install "paddlepaddle-gpu==3.3.1" `
                --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/ --quiet
        }
        else {
            Write-QaiStep 'installing CPU-only paddlepaddle 3.3.1...'
            & $python -m pip install "paddlepaddle==3.3.1" --quiet
        }

        if ($LASTEXITCODE -ne 0) { throw 'failed to install paddlepaddle' }
        Write-QaiOk 'paddlepaddle installed successfully'
    }

    # ------------------------------------------------------------ 3. PaddleOCR Git Checkout
    if ($SkipPaddleOcr) {
        Write-QaiStep 'skipping PaddleOCR checkout (-SkipPaddleOcr)'
    }
    elseif (Test-Path (Join-Path $paddleRoot '.git')) {
        $current = (& git -C $paddleRoot rev-parse HEAD 2>$null)
        if ($current -eq $commit) {
            Write-QaiOk "PaddleOCR already at pinned commit $commit"
        }
        else {
            Write-QaiStep "PaddleOCR is at $current; checking out pinned commit $commit"
            & git -C $paddleRoot fetch --all --tags 2>$null
            & git -C $paddleRoot checkout $commit
            if ($LASTEXITCODE -ne 0) { throw "failed to check out $commit" }
            Write-QaiOk "PaddleOCR pinned to $commit"
        }
    }
    else {
        Write-QaiStep "cloning PaddleOCR repository from $repo..."
        & git clone $repo $paddleRoot
        if ($LASTEXITCODE -ne 0) { throw "failed to clone $repo" }
        & git -C $paddleRoot checkout $commit
        if ($LASTEXITCODE -ne 0) { throw "failed to check out $commit" }
        Write-QaiOk "PaddleOCR cloned and pinned to $commit"
    }

    # ------------------------------------------------------------ 4. Requirements
    Write-QaiStep 'installing unified dependencies (requirements.txt)...'
    & $python -m pip install -r (Join-Path $projectRoot 'requirements.txt') --quiet
    if ($LASTEXITCODE -ne 0) { throw 'failed to install requirements.txt' }
    Write-QaiOk 'requirements.txt installed successfully'
}
finally {
    $ErrorActionPreference = $previousPreference
}

# ---------------------------------------------------------------- 5. Base Models
Write-QaiStep 'verifying / downloading official base models...'
& $python (Join-Path $PSScriptRoot 'download_base_models.py')
if ($LASTEXITCODE -ne 0) { throw 'base models check/download failed' }
Write-QaiOk 'official base models ready'

# ---------------------------------------------------------------- 6. Dataset Download & Prep
if ($SkipDataset) {
    Write-QaiStep 'skipping dataset download and preparation (-SkipDataset)'
}
else {
    $prepared = Join-Path $projectRoot 'data\kurdish_rec\train_rec.txt'
    if (Test-Path $prepared) {
        Write-QaiOk 'prepared dataset already present'
    }
    else {
        Write-QaiStep 'downloading and preparing dataset from Hugging Face...'
        & $python (Join-Path $PSScriptRoot 'prepare_dataset.py') --skip-if-exists
        if ($LASTEXITCODE -ne 0) { throw 'prepare_dataset.py failed' }
        Write-QaiOk 'dataset prepared'
    }

    Write-QaiStep 'auditing prepared dataset...'
    & $python (Join-Path $PSScriptRoot 'audit_dataset.py')
    if ($LASTEXITCODE -ne 0) { throw 'audit_dataset.py failed; inspect the output report above' }
    Write-QaiOk 'dataset audit passed'
}

# ---------------------------------------------------------------- 7. Deep Verification
Write-QaiHeader 'Verifying Environment'
$verifyArgs = @((Join-Path $PSScriptRoot 'verify_env.py'))
if (-not $SkipDataset) { $verifyArgs += '--require-data' }
& $python @verifyArgs
if ($LASTEXITCODE -ne 0) { throw 'environment verification failed' }

$hardware = Get-QaiHardwareProfile
Write-Host ''
Write-QaiHeader 'Setup Complete & Ready'
Write-Host "  GPU Compute : $($hardware.GpuName) ($($hardware.GpuMemoryMB) MB)" -ForegroundColor Green
Write-Host "  CPU / RAM   : $($hardware.CpuCores) cores / $($hardware.RamGB) GB" -ForegroundColor DarkGray
Write-Host "  Batch Size  : $($hardware.RecommendedBatch) (Train) / $($hardware.RecommendedEvalBatch) (Eval)" -ForegroundColor DarkCyan
Write-Host "  Workers     : $($hardware.RecommendedWorkers)" -ForegroundColor DarkCyan
Write-Host ''
Write-Host '  Suggested Workflow Commands:' -ForegroundColor Cyan
Write-Host '    python main.py smoke-test                           # Verify GPU, VRAM & gradient pass' -ForegroundColor White
Write-Host '    python main.py benchmark-base --max-samples 100     # Zero-shot baseline benchmark' -ForegroundColor White
Write-Host '    python main.py pilot-run --num-samples 500          # Rapid 2-epoch mini convergence test' -ForegroundColor White
Write-Host '    python main.py train --epochs 40                    # Full production fine-tuning' -ForegroundColor White
Write-Host '    python main.py export                               # Export trained model for deployment' -ForegroundColor White
Write-Host '    python main.py infer --split test --count 5         # Run inference on test samples' -ForegroundColor White
Write-Host ''
exit 0
