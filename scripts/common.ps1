# Shared helpers for the Kurdish OCR pipeline scripts.
# Dot-source this file:  . (Join-Path $PSScriptRoot 'common.ps1')

Set-StrictMode -Version Latest

function Get-QaiProjectRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Get-QaiPython {
    $python = Join-Path (Get-QaiProjectRoot) '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        throw "Virtual environment missing. Run scripts\setup.ps1 first (expected $python)."
    }
    return $python
}

function Get-QaiPaddleRoot {
    return (Join-Path (Get-QaiProjectRoot) 'PaddleOCR')
}

function Get-QaiNvidiaGpu {
    return (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" -or $_.Caption -like "*NVIDIA*" } | Select-Object -First 1)
}

function Get-QaiNvidiaSmi {
    $command = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidates = @(
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
        'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $driverStore = Get-ChildItem "C:\Windows\System32\DriverStore\FileRepository\*\nvidia-smi.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($driverStore) { return $driverStore.FullName }
    return $null
}

<#
    Picks dataloader workers and train/eval batch size from the actual machine so a
    fresh clone uses the whole GPU and RAM instead of a hard-coded guess.
#>
function Get-QaiHardwareProfile {
    $cpu = [int](Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty NumberOfLogicalProcessors)
    $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    $freeRamGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)

    $gpuName = 'CPU only'
    $gpuMemoryMB = 0
    $smi = Get-QaiNvidiaSmi
    if ($smi) {
        $raw = & $smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($raw) {
            $parts = $raw -split ','
            $gpuName = $parts[0].Trim()
            $gpuMemoryMB = [int]$parts[1].Trim()
        }
    }
    if ($gpuMemoryMB -le 0) {
        $wmiGpu = Get-QaiNvidiaGpu
        if ($wmiGpu) {
            $gpuName = $wmiGpu.Name
            $gpuMemoryMB = 6144  # Safe default estimate for detected modern NVIDIA GPU
        }
    }

    if ($gpuMemoryMB -le 0) {
        $batch = 16
    }
    else {
        $batch = [int][math]::Floor(($gpuMemoryMB * 0.75) / 11.6)
    }
    $batch = [math]::Max(16, ([math]::Floor($batch / 16) * 16))
    $evalBatch = [math]::Min(96, $batch)

    # More loader workers than the CPU count stops helping once the disk read saturates
    # (measured: 8 workers beat 12, which only added contention, and at batch 384
    # avg_reader_cost fell to 0.005 s, i.e. the queue was never even close to empty).
    # Memory is the binding constraint instead: each worker keeps a prefetch queue of
    # full batches, so budget ~1.5 GB per worker against free RAM.
    $cpuCap = [math]::Min(8, [math]::Max(2, [int][math]::Floor($cpu / 1.5)))
    $ramCap = [math]::Max(2, [int][math]::Floor($freeRamGB / 1.5))
    $workers = [math]::Min($cpuCap, $ramCap)

    return [pscustomobject]@{
        CpuCores               = $cpu
        RamGB                  = $ramGB
        FreeRamGB              = $freeRamGB
        GpuName                = $gpuName
        GpuMemoryMB            = $gpuMemoryMB
        RecommendedBatch       = $batch
        RecommendedEvalBatch   = $evalBatch
        RecommendedWorkers     = $workers
    }
}

function Write-QaiHeader {
    param([string]$Title)
    Write-Host ''
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ('=' * 74) -ForegroundColor DarkCyan
}

function Write-QaiStep {
    param([string]$Message)
    Write-Host "  -> $Message" -ForegroundColor DarkGray
}

function Write-QaiOk {
    param([string]$Message)
    Write-Host "  OK  $Message" -ForegroundColor Green
}

function Write-QaiFail {
    param([string]$Message)
    Write-Host "  FAIL $Message" -ForegroundColor Red
}

<#
    Runs the environment verifier and throws if anything fatal is wrong, so no training
    run is ever started on a machine that is known to be misconfigured.
#>
function Assert-QaiEnvironment {
    param([switch]$RequireData, [switch]$RequireCuda)
    $python = Get-QaiPython
    $verifyArgs = @((Join-Path $PSScriptRoot 'verify_env.py'), '--json')
    if ($RequireData) { $verifyArgs += '--require-data' }
    if ($RequireCuda) { $verifyArgs += '--require-cuda' }

    # Native tools such as Paddle write progress notes to stderr; under
    # ErrorActionPreference=Stop that would be raised as a terminating error even on
    # success. Only the exit code decides pass/fail.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $python @verifyArgs 2>&1 | Where-Object { $_ -is [string] -or $_ -isnot [System.Management.Automation.ErrorRecord] }
    }
    finally {
        $ErrorActionPreference = $previous
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host ($output | Out-String) -ForegroundColor DarkGray
        throw 'Environment verification failed. Fix the problems above, or run scripts\setup.ps1.'
    }
    return ($output | Out-String | ConvertFrom-Json)
}
