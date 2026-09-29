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
else {
    # Check MSVC runtime on Windows
    $vcRuntime = Join-Path $env:SystemRoot 'System32\vcruntime140.dll'
    if (-not (Test-Path $vcRuntime)) {
        Write-Host "  [!] Microsoft Visual C++ 2015-2022 Redistributable (x64) is missing." -ForegroundColor Yellow
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            Write-QaiStep "Installing Visual C++ Redistributable via winget..."
            & winget install Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements 2>$null
        }
    }

    # Enable Git long paths to prevent path length issues
    & git config --global core.longpaths true 2>$null

    if (Test-Path $venvPython) {
        $existingVer = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{sys.maxsize > 2**32}')" 2>$null)
        if ($existingVer) {
            $parts = $existingVer.Trim().Split('|')
            $pyVer = $parts[0].Trim()
            $is64 = ($parts[1].Trim() -eq "True")
            $bitText = if ($is64) { '64-bit' } else { '32-bit' }
            if ($pyVer -ge "3.13" -or (-not $is64)) {
                Write-QaiStep "Existing .venv uses incompatible Python $pyVer ($bitText). PaddlePaddle requires 64-bit Python 3.10-3.12."
                Write-QaiStep "Removing incompatible .venv..."
                Remove-Item -Recurse -Force $venv
            } else {
                Write-QaiOk "virtualenv present (Python $pyVer $bitText)"
            }
        }
    }

    if (-not (Test-Path $venvPython)) {
        Write-QaiStep 'creating virtual environment .venv...'
        $found = $false
        foreach ($ver in @('3.12', '3.11', '3.10')) {
            $probe = & py -$ver -c "import sys; print(sys.maxsize > 2**32)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe.Trim() -eq "True") {
                Write-QaiStep "Creating .venv using 64-bit Python $ver..."
                & py -$ver -m venv $venv
                $found = $true
                break
            }
        }
        if (-not $found) {
            $defCheck = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{sys.maxsize > 2**32}')" 2>$null
            if ($defCheck) {
                $parts = $defCheck.Trim().Split('|')
                if ($parts[0].Trim() -lt "3.13" -and $parts[1].Trim() -eq "True") {
                    & python -m venv $venv
                    $found = $true
                }
            }
        }
        if (-not $found) {
            $winget = Get-Command winget -ErrorAction SilentlyContinue
            if ($winget) {
                Write-Host "  -> Compatible 64-bit Python 3.10-3.12 not found on this PC." -ForegroundColor Yellow
                Write-Host "  -> Attempting automated installation of Python 3.12 via winget..." -ForegroundColor Cyan
                & winget install Python.Python.3.12 --scope currentuser --silent --accept-package-agreements --accept-source-agreements
                $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
                $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
                $env:Path = "$userPath;$machinePath"

                $probe = & py -3.12 -c "import sys; print(sys.maxsize > 2**32)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $probe.Trim() -eq "True") {
                    Write-QaiStep "Creating .venv using freshly installed 64-bit Python 3.12..."
                    & py -3.12 -m venv $venv
                    $found = $true
                }
            }
        }
        if (-not (Test-Path $venvPython)) {
            Write-Host ""
            Write-Host "========================================================================" -ForegroundColor Red
            Write-Host " [FATAL] Compatible 64-bit Python 3.10-3.12 is required" -ForegroundColor Red
            Write-Host "========================================================================" -ForegroundColor Red
            Write-Host " PaddlePaddle 3.3.1 currently supports 64-bit Python 3.10, 3.11, or 3.12." -ForegroundColor White
            Write-Host " Python 3.13 and 32-bit versions are not yet supported by PaddlePaddle." -ForegroundColor Yellow
            Write-Host ""
            Write-Host " Quick Fix: Open PowerShell and run:" -ForegroundColor Cyan
            Write-Host "   winget install Python.Python.3.12" -ForegroundColor Green
            Write-Host " Or download from: https://www.python.org/downloads/release/python-3129/" -ForegroundColor White
            Write-Host "========================================================================" -ForegroundColor Red
            throw "PaddlePaddle requires 64-bit Python 3.10-3.12"
        }
        Write-QaiOk 'created .venv'
    }
}

if (-not (Test-Path $venvPython)) { throw "no interpreter at $venvPython" }
$python = $venvPython

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'

try {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    $venvUv = Join-Path $venv 'Scripts\uv.exe'
    if (-not (Test-Path $venvUv)) {
        $altUv = Join-Path $venv 'bin\uv'
        if (Test-Path $altUv) { $venvUv = $altUv }
    }

    if (-not $uv -and (Test-Path $venvUv)) {
        $uv = Get-Command $venvUv -ErrorAction SilentlyContinue
    }

    if (-not $uv) {
        Write-QaiStep 'ensuring uv package manager in virtual environment...'
        & $python -m pip install uv --quiet
        if (Test-Path $venvUv) {
            $uv = Get-Command $venvUv -ErrorAction SilentlyContinue
        }
    }

    if ($uv) {
        $uvVer = & $uv.Source --version 2>$null
        Write-QaiOk "uv package manager ready ($uvVer) - utilizing ultra-fast caching"
    } else {
        Write-QaiStep 'upgrading pip and setuptools...'
        & $python -m pip install --upgrade pip setuptools wheel --timeout 120 --retries 5 --quiet
        if ($LASTEXITCODE -ne 0) { throw 'failed to upgrade pip' }
        Write-QaiOk 'pip ready'
    }

    # ------------------------------------------------------------ 2. PaddlePaddle Installation
    if ($SkipPaddleInstall) {
        Write-QaiStep 'skipping Paddle install (-SkipPaddleInstall)'
    }
    else {
        $nvidiaGpu = Get-QaiNvidiaGpu
        $smi = Get-QaiNvidiaSmi
        $hasNvidia = ($null -ne $nvidiaGpu) -or ($null -ne $smi)
        $gpuTitle = if ($nvidiaGpu) { $nvidiaGpu.Name } else { "NVIDIA GPU" }
        $cudaVer = 0.0

        if ($smi) {
            $smiOut = & $smi 2>$null
            if ($smiOut -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') {
                $cudaVer = [double]$matches[1]
            }
        }

        $isBlackwellOr50 = ($gpuTitle -match "RTX 50|Blackwell") -or ($cudaVer -ge 12.8) -or ($cudaVer -ge 13.0)
        $installedGpu = $false

        if ($hasNvidia -and (-not $CpuOnly)) {
            # Remove CPU paddlepaddle to avoid namespace collision
            & $python -m pip uninstall -y paddlepaddle 2>$null

            if ($isBlackwellOr50) {
                Write-QaiStep "Detected Blackwell / RTX 50-Series architecture ($gpuTitle, CUDA $cudaVer) -> installing paddlepaddle-gpu 3.4.0 (cu129)..."
                & $python -m pip install "paddlepaddle-gpu==3.4.0" `
                    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --timeout 120 --retries 5
                if ($LASTEXITCODE -eq 0) { $installedGpu = $true }
            }

            if (-not $installedGpu) {
                if ($cudaVer -lt 12.0 -and $cudaVer -ge 11.0) {
                    Write-QaiStep "Detected $gpuTitle (CUDA $cudaVer) -> installing paddlepaddle-gpu 3.3.1 (cu118)..."
                    & $python -m pip install "paddlepaddle-gpu==3.3.1" `
                        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/ --timeout 120 --retries 5
                    if ($LASTEXITCODE -eq 0) { $installedGpu = $true }
                } else {
                    Write-QaiStep "Detected $gpuTitle (CUDA $cudaVer) -> installing paddlepaddle-gpu 3.3.1 (cu126)..."
                    & $python -m pip install "paddlepaddle-gpu==3.3.1" `
                        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --timeout 120 --retries 5
                    if ($LASTEXITCODE -eq 0) { $installedGpu = $true }
                    if (-not $installedGpu) {
                        Write-QaiStep "cu126 installation retry with cu118..."
                        & $python -m pip install "paddlepaddle-gpu==3.3.1" `
                            --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/ --timeout 120 --retries 5
                        if ($LASTEXITCODE -eq 0) { $installedGpu = $true }
                    }
                }
            }

            if ($installedGpu) {
                Write-QaiOk "paddlepaddle-gpu installed successfully for $gpuTitle"
            } else {
                Write-Host "  [!] GPU wheel installation failed. Falling back cleanly to CPU..." -ForegroundColor Yellow
            }
        }

        if (-not $installedGpu) {
            Write-QaiStep 'installing CPU-only paddlepaddle 3.3.1...'
            & $python -m pip uninstall -y paddlepaddle-gpu 2>$null
            & $python -m pip install "paddlepaddle==3.3.1" --timeout 120 --retries 5
            if ($LASTEXITCODE -ne 0) { throw 'failed to install paddlepaddle' }
            Write-QaiOk 'paddlepaddle installed successfully (CPU mode)'
        }
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

    # Apply required runtime patches to PaddleOCR
    Write-QaiStep 'applying Kurdish pipeline patches to PaddleOCR...'
    & $python (Join-Path $PSScriptRoot 'patch_paddleocr.py')
    Write-QaiOk 'PaddleOCR patches verified'

    # ------------------------------------------------------------ 4. Requirements
    Write-QaiStep 'installing unified dependencies (requirements.txt)...'
    if ($uv) {
        & $uv.Source pip install --python $python -r (Join-Path $projectRoot 'requirements.txt')
    } else {
        & $python -m pip install -r (Join-Path $projectRoot 'requirements.txt') --quiet
    }
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
Write-Host "  CPU / RAM   : $($hardware.CpuCores) cores / $($hardware.RamGB) GB ($($hardware.ReservedFreeRamGB) GB reserved free)" -ForegroundColor DarkGray
Write-Host "  Batch Size  : $($hardware.RecommendedBatch) (Train) / $($hardware.RecommendedEvalBatch) (Eval)" -ForegroundColor DarkCyan
Write-Host "  Workers     : $($hardware.RecommendedWorkers) (Memory & thread safety capped)" -ForegroundColor DarkCyan
Write-Host ''
Write-Host '  Suggested Workflow Commands:' -ForegroundColor Cyan
Write-Host '    python main.py serve                                # Launch interactive web studio' -ForegroundColor Yellow
Write-Host '    python main.py smoke-test                           # Verify GPU, VRAM & gradient pass' -ForegroundColor White
Write-Host '    python main.py benchmark-base --max-samples 100     # Zero-shot baseline benchmark' -ForegroundColor White
Write-Host '    python main.py pilot-run --num-samples 500          # Rapid 2-epoch mini convergence test' -ForegroundColor White
Write-Host '    python main.py train --epochs 10                    # Full production fine-tuning' -ForegroundColor White
Write-Host '    python main.py export                               # Export trained model for deployment' -ForegroundColor White
Write-Host '    python main.py infer --split test --count 5         # Run inference on test samples' -ForegroundColor White
Write-Host ''
exit 0
