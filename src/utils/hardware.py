"""Hardware profiling, memory estimation, and dynamic resource allocation.

Detects available GPU/CPU resources, enforces user-defined host RAM safety guards
(defaulting to strictly preserving 15.0 GB free RAM), and offers interactive or automated
selection of optimal training batch sizes, DataLoader workers, and memory-pinned DMA pipelines.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware capabilities and calculated runtime configurations."""

    has_gpu: bool
    gpu_name: str
    gpu_memory_mb: int
    cpu_cores: int
    ram_gb: float
    free_ram_gb: float
    reserved_free_ram_gb: float
    usable_ram_gb: float
    recommended_train_batch: int
    recommended_eval_batch: int
    recommended_workers: int
    recommended_amp_level: str


@dataclass
class ResourcePreset:
    """A selectable training resource profile."""

    key: int
    name: str
    description: str
    train_batch: int
    eval_batch: int
    workers: int
    pin_memory: bool
    est_ram_gb: float
    projected_free_ram_gb: float
    is_recommended: bool = False


def estimate_training_memory(batch_size: int, workers: int, free_ram_gb: float) -> tuple[float, float]:
    """Estimate host RAM footprint and projected free RAM during PaddleOCR training.

    Args:
        batch_size: Training batch size per card.
        workers: DataLoader worker process count.
        free_ram_gb: Current available free host RAM.

    Returns:
        tuple of (estimated_training_ram_gb, projected_free_ram_gb)
    """
    # Base Python process, CUDA runtime buffers, Paddle OCR graph, and dataset index: ~2.5 GB
    # Each DataLoader worker process with sample decoding queue: ~350 MB (0.35 GB)
    # Batch buffer queue memory: ~4 MB per batch sample (0.004 GB)
    est_ram = round(2.5 + (workers * 0.35) + (batch_size * 0.004), 1)
    projected_free = max(0.0, round(free_ram_gb - est_ram, 1))
    return est_ram, projected_free


def get_hardware_profile(reserved_ram_gb: Optional[float] = None) -> HardwareProfile:
    """Detect available compute hardware and compute safe, high-throughput hyperparameter defaults.

    Args:
        reserved_ram_gb: Optional amount of host RAM (in GB) to strictly preserve free.
                         Defaults to 15.0 GB on >=24 GB systems to honor user safety requirements.

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
                gpu_memory_mb = 8192
        except Exception:
            pass

    # Recommended batch sizing tuned for PP-OCRv5 recognition (SVTR backbone + CTC + NRTR MultiHead):
    if not has_gpu or gpu_memory_mb <= 0:
        recommended_train_batch = 32
        recommended_eval_batch = 16
        recommended_amp = "O0"
    else:
        if gpu_memory_mb <= 4096:
            recommended_train_batch = 64
        elif gpu_memory_mb <= 6144:
            recommended_train_batch = 128
        elif gpu_memory_mb <= 8192:
            recommended_train_batch = 384
        elif gpu_memory_mb <= 12288:
            recommended_train_batch = 512
        else:
            recommended_train_batch = 512
        recommended_eval_batch = min(256, recommended_train_batch)
        recommended_amp = "O2"

    # Host RAM preservation logic:
    # Strictly enforce preserving 15.0 GB free RAM for the OS & background tasks,
    # reserving the entire remainder for training and dataloader worker processes.
    if reserved_ram_gb is not None:
        reserved_free_ram_gb = max(1.0, min(float(reserved_ram_gb), ram_gb - 2.0))
    elif ram_gb >= 24.0:
        reserved_free_ram_gb = 15.0
    elif ram_gb >= 16.0:
        reserved_free_ram_gb = 4.0
    else:
        reserved_free_ram_gb = 2.0

    usable_ram_gb = max(0.5, round(free_ram_gb - reserved_free_ram_gb, 1))

    # DataLoader workers scaling:
    # Scale aggressively for multi-core / high-RAM workstations without choking.
    ram_worker_cap = max(1, int(usable_ram_gb // 0.35))
    if cpu_cores >= 32:
        target_workers = min(24, max(4, int(cpu_cores * 0.5)))
    elif cpu_cores >= 16:
        target_workers = min(16, max(4, int(cpu_cores * 0.5)))
    elif cpu_cores >= 8:
        target_workers = min(12, max(2, int(cpu_cores * 0.5)))
    elif cpu_cores >= 4:
        target_workers = min(4, max(1, cpu_cores // 2))
    else:
        target_workers = 2

    recommended_workers = max(1, min(target_workers, ram_worker_cap))

    return HardwareProfile(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        free_ram_gb=free_ram_gb,
        reserved_free_ram_gb=reserved_free_ram_gb,
        usable_ram_gb=usable_ram_gb,
        recommended_train_batch=recommended_train_batch,
        recommended_eval_batch=recommended_eval_batch,
        recommended_workers=recommended_workers,
        recommended_amp_level=recommended_amp,
    )


def generate_resource_presets(profile: HardwareProfile) -> list[ResourcePreset]:
    """Generate curated resource presets (Max, Balanced, Conservative) for a hardware profile."""
    # Preset 1: Max Performance (Recommended)
    p1_tb = profile.recommended_train_batch
    p1_eb = profile.recommended_eval_batch
    p1_w = profile.recommended_workers
    p1_est, p1_proj = estimate_training_memory(p1_tb, p1_w, profile.free_ram_gb)
    preset1 = ResourcePreset(
        key=1,
        name="High Throughput / Max Performance (RECOMMENDED)",
        description="Max GPU saturation and high worker parallelism utilizing all left RAM while preserving 15 GB free",
        train_batch=p1_tb,
        eval_batch=p1_eb,
        workers=p1_w,
        pin_memory=True,
        est_ram_gb=p1_est,
        projected_free_ram_gb=p1_proj,
        is_recommended=True,
    )

    # Preset 2: Balanced Workload
    p2_tb = max(32, int(p1_tb * 0.65) // 32 * 32 or 64)
    p2_eb = max(16, int(p1_eb * 0.65) // 32 * 32 or 32)
    p2_w = max(2, p1_w // 2)
    p2_est, p2_proj = estimate_training_memory(p2_tb, p2_w, profile.free_ram_gb)
    preset2 = ResourcePreset(
        key=2,
        name="Balanced Workload",
        description="Moderate batch size and worker pool for quiet operation alongside active desktop tasks",
        train_batch=p2_tb,
        eval_batch=p2_eb,
        workers=p2_w,
        pin_memory=True,
        est_ram_gb=p2_est,
        projected_free_ram_gb=p2_proj,
    )

    # Preset 3: Conservative / Low RAM Overhead
    p3_tb = max(32, int(p1_tb * 0.35) // 32 * 32 or 32)
    p3_eb = max(16, int(p1_eb * 0.35) // 16 * 16 or 16)
    p3_w = max(1, p1_w // 4)
    p3_est, p3_proj = estimate_training_memory(p3_tb, p3_w, profile.free_ram_gb)
    preset3 = ResourcePreset(
        key=3,
        name="Conservative / Low RAM Overhead",
        description="Lightweight memory footprint maximizing free system headroom",
        train_batch=p3_tb,
        eval_batch=p3_eb,
        workers=p3_w,
        pin_memory=True,
        est_ram_gb=p3_est,
        projected_free_ram_gb=p3_proj,
    )

    return [preset1, preset2, preset3]


def _safe_print(text: str) -> None:
    """Print text safely across legacy Windows consoles without UnicodeEncodeError."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        except Exception:
            print(text.encode("ascii", errors="replace").decode("ascii"))


def configure_training_resources(
    args: Any,
    is_pipeline: bool = False,
) -> tuple[int, int, int, bool]:
    """Interactively or automatically configure batch size, workers, and memory reserves.

    Enforces user requirement to strictly maintain 15 GB of host RAM free (or user-specified
    --reserve-ram-gb), allocating all remaining RAM and CPU resources to maximize training throughput.

    Returns:
        tuple of (train_batch_size, eval_batch_size, num_workers, pin_memory)
    """
    reserve_target = getattr(args, "reserve_ram_gb", 15.0)
    profile = get_hardware_profile(reserved_ram_gb=reserve_target)
    presets = generate_resource_presets(profile)
    p1 = presets[0]

    explicit_bs = getattr(args, "batch_size", None)
    explicit_workers = getattr(args, "workers", None)
    force_interactive = getattr(args, "interactive", False)
    no_interactive = getattr(args, "no_interactive", False)

    is_interactive_terminal = sys.stdin.isatty() and not no_interactive
    should_prompt = force_interactive or (is_interactive_terminal and (explicit_bs is None or explicit_workers is None))

    chosen_bs = explicit_bs or p1.train_batch
    chosen_eval_bs = getattr(args, "eval_batch_size", None) or min(chosen_bs, p1.eval_batch)
    chosen_workers = explicit_workers if explicit_workers is not None else p1.workers
    chosen_pin = not getattr(args, "no_pin_memory", False)
    selection_label = "Auto-Tuned Recommended Profile"

    if should_prompt:
        border = "=" * 78
        divider = "-" * 78
        _safe_print("\n" + border)
        _safe_print(" [HARDWARE] KURDISH OCR HARDWARE & MEMORY ALLOCATION CONFIGURATOR")
        _safe_print(border)
        _safe_print("  System Hardware Detected:")
        _safe_print(f"    * Active Compute:    {profile.gpu_name} ({profile.gpu_memory_mb} MB VRAM)")
        _safe_print(f"    * CPU Architecture:  {profile.cpu_cores} Cores")
        _safe_print(f"    * Total System RAM:  {profile.ram_gb:.1f} GB")
        _safe_print(f"    * Current Free RAM:  {profile.free_ram_gb:.1f} GB")
        _safe_print(f"    * OS Safety Guard:   {profile.reserved_free_ram_gb:.1f} GB (Preserved strictly free for OS & Desktop)")
        _safe_print(f"    * Usable for Train:  {profile.usable_ram_gb:.1f} GB (All remaining RAM devoted to OCR training)")
        _safe_print(divider)
        _safe_print("  Select Resource Preset Profile for Training:\n")

        for p in presets:
            rec_tag = " [RECOMMENDED]" if p.is_recommended else ""
            _safe_print(f"  [{p.key}] {p.name}{rec_tag}")
            _safe_print(f"      * Train Batch: {p.train_batch:<4} | Eval Batch: {p.eval_batch:<4} | DataLoader Workers: {p.workers}")
            _safe_print(f"      * Est. RAM: ~{p.est_ram_gb:.1f} GB | Projected Free RAM: ~{p.projected_free_ram_gb:.1f} GB (Safety: >= {profile.reserved_free_ram_gb:.1f} GB)")
            _safe_print(f"      * {p.description}\n")

        _safe_print("  [4] Custom Manual Configuration")
        _safe_print("      * Specify custom Batch Size, Worker Count, and Pinned Memory directly\n")
        _safe_print(divider)

        try:
            choice = input(f"  Select option [1-4] (Press Enter for [1] Recommended): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "1"
            _safe_print("1")

        if choice in ("", "1"):
            chosen_bs = p1.train_batch
            chosen_eval_bs = p1.eval_batch
            chosen_workers = p1.workers
            chosen_pin = True
            selection_label = "Preset [1] - High Throughput / Max Performance (Recommended)"
        elif choice == "2":
            p2 = presets[1]
            chosen_bs = p2.train_batch
            chosen_eval_bs = p2.eval_batch
            chosen_workers = p2.workers
            chosen_pin = True
            selection_label = "Preset [2] - Balanced Workload"
        elif choice == "3":
            p3 = presets[2]
            chosen_bs = p3.train_batch
            chosen_eval_bs = p3.eval_batch
            chosen_workers = p3.workers
            chosen_pin = True
            selection_label = "Preset [3] - Conservative / Low RAM Overhead"
        elif choice == "4":
            selection_label = "Preset [4] - Custom Manual Configuration"
            try:
                raw_bs = input(f"    -> Enter Train Batch Size [default {p1.train_batch}]: ").strip()
                chosen_bs = int(raw_bs) if raw_bs.isdigit() and int(raw_bs) > 0 else p1.train_batch
            except Exception:
                chosen_bs = p1.train_batch

            try:
                raw_w = input(f"    -> Enter DataLoader Workers [default {p1.workers}]: ").strip()
                chosen_workers = int(raw_w) if raw_w.isdigit() and int(raw_w) >= 0 else p1.workers
            except Exception:
                chosen_workers = p1.workers

            try:
                raw_pin = input("    -> Enable Pinned Memory? (Y/n) [default Y]: ").strip().lower()
                chosen_pin = raw_pin not in ("n", "no", "false", "0")
            except Exception:
                chosen_pin = True

            chosen_eval_bs = min(chosen_bs, p1.eval_batch)
        else:
            _safe_print(f"  Unrecognized option '{choice}'. Defaulting to [1] Recommended.")
            chosen_bs = p1.train_batch
            chosen_eval_bs = p1.eval_batch
            chosen_workers = p1.workers
            chosen_pin = True
            selection_label = "Preset [1] - High Throughput / Max Performance (Recommended)"

    elif explicit_bs is not None and explicit_workers is not None:
        selection_label = "CLI Explicitly Specified Hyperparameters"

    # Compute final memory footprint with selected parameters
    est_ram, proj_free = estimate_training_memory(chosen_bs, chosen_workers, profile.free_ram_gb)
    safe_margin = proj_free - profile.reserved_free_ram_gb
    if safe_margin >= 0:
        margin_status = f"[SAFE] (+{safe_margin:.1f} GB buffer above {profile.reserved_free_ram_gb:.1f} GB floor)"
    else:
        margin_status = f"[ATTENTION] ({abs(safe_margin):.1f} GB below {profile.reserved_free_ram_gb:.1f} GB target)"

    # Print summary block
    _safe_print("\n" + "=" * 78)
    _safe_print(" [RESOURCE ALLOCATION & MEMORY BREAKDOWN]")
    _safe_print("=" * 78)
    _safe_print(f"  * Configuration Mode:    {selection_label}")
    _safe_print(f"  * Total System RAM:      {profile.ram_gb:.1f} GB")
    _safe_print(f"  * Reserved OS Guard:     {profile.reserved_free_ram_gb:.1f} GB (Strictly Protected Free Space)")
    _safe_print(f"  * Usable for Training:   {profile.usable_ram_gb:.1f} GB")
    _safe_print(f"  * Est. Training RAM:     ~{est_ram:.1f} GB (Base Engine + {chosen_workers} Workers + Buffers)")
    _safe_print(f"  * Projected Free RAM:    ~{proj_free:.1f} GB ({margin_status})")
    _safe_print(f"  * Training Batch Size:   {chosen_bs} per GPU card")
    _safe_print(f"  * Validation Batch Size: {chosen_eval_bs} per GPU card")
    _safe_print(f"  * DataLoader Workers:    {chosen_workers} parallel processes")
    _safe_print(f"  * Pinned Memory (DMA):   {'Enabled (Page-locked GPU direct transfer)' if chosen_pin else 'Disabled'}")
    _safe_print(f"  * Precision Mode:        {profile.recommended_amp_level}")
    _safe_print("=" * 78 + "\n")

    # Update args namespace for downstream consistency
    args.batch_size = chosen_bs
    args.eval_batch_size = chosen_eval_bs
    args.workers = chosen_workers
    args.no_pin_memory = not chosen_pin

    return chosen_bs, chosen_eval_bs, chosen_workers, chosen_pin
