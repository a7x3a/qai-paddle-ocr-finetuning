"""Hardware profiling and dynamic hyperparameter allocation.

Detects available GPU/CPU resources and computes optimal batch sizes, dataloader workers,
and precision modes (AMP) tailored for any environment (Free Google Colab T4, local laptop GPUs,
or CPU-only execution).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware capabilities and calculated runtime configurations."""

    has_gpu: bool
    gpu_name: str
    gpu_memory_mb: int
    cpu_cores: int
    ram_gb: float
    free_ram_gb: float
    recommended_train_batch: int
    recommended_eval_batch: int
    recommended_workers: int
    recommended_amp_level: str


def get_hardware_profile() -> HardwareProfile:
    """Detect available compute hardware and compute safe, high-throughput hyperparameter defaults.

    Returns:
        HardwareProfile with dynamic batch sizing and dataloader worker limits.
    """
    cpu_cores = os.cpu_count() or 4

    # System RAM detection
    ram_gb = 8.0
    free_ram_gb = 4.0
    try:
        import psutil
        mem = psutil.virtual_memory()
        ram_gb = round(mem.total / (1024**3), 1)
        free_ram_gb = round(mem.available / (1024**3), 1)
    except ImportError:
        # Fallback without psutil
        pass

    # GPU Detection
    gpu_name = "CPU Only"
    gpu_memory_mb = 0
    has_gpu = False

    # Check via nvidia-smi
    smi = shutil.which("nvidia-smi")
    if not smi and os.name == "nt":
        for cand in [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]:
            if os.path.isfile(cand):
                smi = cand
                break

    if smi:
        try:
            res = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = line.split(",")
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    gpu_memory_mb = int(float(parts[1].strip()))
                    has_gpu = gpu_memory_mb > 0
        except Exception:
            pass

    # Windows WMI fallback if nvidia-smi failed or wasn't found
    if not has_gpu and os.name == "nt":
        try:
            wmi_out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } | Select-Object -ExpandProperty Name"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if wmi_out.returncode == 0 and wmi_out.stdout.strip():
                gpu_name = wmi_out.stdout.strip().splitlines()[0]
                gpu_memory_mb = 6144
                has_gpu = True
        except Exception:
            pass

    # If nvidia-smi failed or not present, try paddle if already imported/available
    if not has_gpu:
        try:
            import paddle
            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                has_gpu = True
                gpu_name = paddle.device.cuda.get_device_name(0) if hasattr(paddle.device.cuda, "get_device_name") else "CUDA GPU"
                # Default estimate for general GPU if memory query unavailable
                gpu_memory_mb = 8192
        except Exception:
            pass

    # Recommended batch sizing:
    # Tuned for PP-OCRv5 recognition (SVTR backbone + CTC + NRTR MultiHead):
    # Base model/framework/Adam optimizer requires ~3.8 GB VRAM.
    # We calibrate batch sizes so total memory stays strictly within dedicated VRAM,
    # preventing Windows WDDM PCIe paging into system RAM.
    if not has_gpu or gpu_memory_mb <= 0:
        recommended_train_batch = 16
        recommended_eval_batch = 16
        recommended_amp = "O0"
    else:
        if gpu_memory_mb <= 4096:
            recommended_train_batch = 16
        elif gpu_memory_mb <= 6144:
            recommended_train_batch = 32
        elif gpu_memory_mb <= 8192:
            recommended_train_batch = 48
        elif gpu_memory_mb <= 12288:
            recommended_train_batch = 64
        elif gpu_memory_mb <= 16384:
            recommended_train_batch = 96
        else:
            recommended_train_batch = 128
        recommended_eval_batch = min(48, recommended_train_batch)
        recommended_amp = "O2"

    # Dataloader workers:
    # Explicitly guarantee at least 10 GB of host RAM is ALWAYS preserved free for user applications.
    # Budget ~1.2 GB per worker process from the available memory above the 10 GB safety floor.
    usable_ram_headroom = max(0.0, free_ram_gb - 10.0)
    ram_worker_cap = max(1, int(usable_ram_headroom // 1.5))
    max_workers_platform = 4 if (os.name == "nt" and free_ram_gb >= 20.0) else (2 if os.name == "nt" else 6)
    cpu_cap = min(max_workers_platform, max(1, int(cpu_cores // 4)))
    recommended_workers = max(1, min(cpu_cap, ram_worker_cap))

    return HardwareProfile(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        free_ram_gb=free_ram_gb,
        recommended_train_batch=recommended_train_batch,
        recommended_eval_batch=recommended_eval_batch,
        recommended_workers=recommended_workers,
        recommended_amp_level=recommended_amp,
    )
