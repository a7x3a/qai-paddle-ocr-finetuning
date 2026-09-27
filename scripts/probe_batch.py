"""Probe trainable batch sizes and report speed, memory, and throughput.

The smoke run showed batch 32 using only ~634 MB of a 6 GB GPU, so the default is far
too small to keep the hardware busy. This runs the real PaddleOCR training entry point
at several batch sizes, lets each one settle, then reports images/second, peak memory and
whether the choice is reader-bound (more workers help) or GPU-bound (only a faster GPU
helps). Out-of-memory is recorded as a result, never a crash.

    python scripts/probe_batch.py                       # default candidates
    python scripts/probe_batch.py --sizes 64,128,256    # explicit candidates
    python scripts/probe_batch.py --subset 4000 --timeout 120
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import logparse
    from lib import reports
    from lib.paths import REC_ROOT, PADDLE_ROOT, TRAIN_CONFIG, REPORT_ROOT, OUTPUT_ROOT
else:  # pragma: no cover
    from .lib import logparse
    from .lib import reports
    from .lib.paths import REC_ROOT, PADDLE_ROOT, TRAIN_CONFIG, REPORT_ROOT, OUTPUT_ROOT

# Every candidate must be a multiple of 16: MultiScaleSampler divides the first batch by
# each entry of divided_factor ([8, 16]) to build the other scales.
DEFAULT_SIZES = (32, 64, 96, 128, 192, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES),
                        help="Comma-separated batch sizes to try")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Dataloader workers. -1 picks a value from the CPU count; "
                             "0 runs in the main process, which on Windows can beat worker "
                             "processes because paddle's spawn + IPC is expensive")
    parser.add_argument("--subset", type=int, default=4000, help="Training samples per candidate")
    parser.add_argument("--timeout", type=int, default=150, help="Seconds allowed per candidate")
    parser.add_argument("--cooldown", type=int, default=20,
                        help="Seconds to idle between candidates so a laptop GPU is not "
                             "penalising later measurements with thermal throttling")
    parser.add_argument("--repeat", type=int, default=1, help="Passes over the candidate list")
    parser.add_argument("--print-step", type=int, default=100,
                        help="Global.print_batch_step. Every N steps PaddleOCR runs an extra "
                             "eval-mode forward pass plus a CPU CTC decode and Levenshtein; at "
                             "1 this dominates the measured step time and makes throughput look "
                             "far worse than a normal run")
    parser.add_argument("--eval-step", type=int, default=0,
                        help="Global.eval_batch_step. Validation with AMP O2 is what exhausted a "
                             "6 GB card during the pilot, so set this to a small value such as 20 "
                             "to make the train+eval memory interaction show up in seconds instead "
                             "of at step 1000. 0 disables validation.")
    parser.add_argument("--eval-batch", type=int, default=96,
                        help="Validation batch size; must stay well below the training batch")
    parser.add_argument("--unfreeze", action="store_true",
                        help="Measure with the backbone trainable from step 0. This is the only "
                             "configuration that matters: with the backbone frozen, batch 384 "
                             "reserves 4365 MB and trains happily, then unfreezing adds the "
                             "backbone gradients and Adam states (~1.9 GB) and immediately "
                             "exhausts a 6 GB card. Probing a frozen backbone understates real "
                             "memory by roughly 40%.")
    parser.add_argument("--amp-level", default="O2", choices=["O0", "O1", "O2"])
    parser.add_argument("--shared-memory", dest="shared_memory", action="store_true", default=True,
                        help="Paddle's file-backed shared-memory loader (default)")
    parser.add_argument("--no-shared-memory", dest="shared_memory", action="store_false",
                        help="Use pipe-based transfer instead; on Windows the shared-memory "
                             "queue is often far slower than plain pipes")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_ROOT)
    return parser.parse_args()


def ensure_subset(size: int) -> str:
    """Write (once) a deterministic head slice of the training list."""
    name = f"train_probe_{size}.txt"
    target = REC_ROOT / name
    if not target.is_file():
        source = REC_ROOT / "train_rec.txt"
        lines = source.read_text(encoding="utf-8").splitlines()
        take = min(size, len(lines))
        target.write_text("\n".join(lines[:take]) + "\n", encoding="utf-8")
    return name


def build_overrides(batch: int, workers: int, label_file: str, output_dir: str,
                    amp_level: str, use_gpu: bool, shared_memory: bool, print_step: int, eval_step: int, eval_batch: int, unfreeze: bool) -> list[str]:
    return [
        f"Global.save_model_dir={output_dir}",
        "Global.pretrained_model=../pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained",
        "Global.epoch_num=1",
        f"Global.use_gpu={str(use_gpu)}",
        f"Global.amp_level={amp_level}",
        f"Global.print_batch_step={print_step}",
        # 0 = backbone trainable from the first step, i.e. measure the real worst case.
        "Global.freeze_backbone_epochs=0" if unfreeze else "Global.freeze_backbone_epochs=1000",
        "Global.save_epoch_step=1000",
        f"Train.sampler.first_bs={batch}",
        f"Train.loader.batch_size_per_card={batch}",
        f"Train.loader.num_workers={workers}",
        f"Train.loader.use_shared_memory={shared_memory}",
        f"Global.eval_batch_step=[0,{eval_step}]" if eval_step else "Global.eval_batch_step=[0,100000000]",
        f"Eval.loader.batch_size_per_card={eval_batch}",
        f"Train.dataset.label_file_list=['../data/kurdish_rec/{label_file}']",
    ]


def run_candidate(batch: int, workers: int, args: argparse.Namespace) -> dict:
    label_file = ensure_subset(args.subset)
    output_name = f"probe_b{batch}"
    output_dir = f"../output/{output_name}"
    # train.py resolves save_model_dir against the PaddleOCR checkout, so the log lands
    # under the project's output/, not one level above it.
    local_output = OUTPUT_ROOT / output_name
    if local_output.exists():
        for stale in local_output.glob("*"):
            stale.unlink()
    local_output.mkdir(parents=True, exist_ok=True)
    log_file = local_output / "train.log"

    overrides = build_overrides(batch, workers, label_file, output_dir, args.amp_level, not args.cpu, args.shared_memory, args.print_step, args.eval_step, args.eval_batch, args.unfreeze)
    environment = dict(os.environ)
    environment["GLOG_minloglevel"] = "2"
    environment["FLAGS_call_stack_level"] = "0"

    record: dict = {"batch_size": batch, "workers": workers, "status": "unknown"}
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "tools/train.py", "-c", str(TRAIN_CONFIG), "-o", *overrides],
        cwd=str(PADDLE_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
        errors="replace",
    )
    try:
        _, stderr = process.communicate(timeout=args.timeout)
        record["status"] = "completed" if process.returncode == 0 else f"exit_{process.returncode}"
        record["stderr_tail"] = (stderr or "")[-600:]
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        record["status"] = "timed_out"
    record["wall_seconds"] = round(time.monotonic() - started, 1)

    if log_file.is_file():
        parsed = logparse.parse_log(log_file)
        summary = logparse.summarise(parsed)
        record["steady_state_ips"] = summary.get("steady_state_ips")
        record["peak_gpu_reserved_mb"] = summary.get("peak_gpu_reserved_mb")
        record["peak_gpu_allocated_mb"] = summary.get("peak_gpu_allocated_mb")
        record["avg_reader_cost"] = summary.get("avg_reader_cost")
        record["avg_batch_cost"] = summary.get("avg_batch_cost")
        record["batches_logged"] = len(parsed.get("batches", []))

    stderr_text = record.get("stderr_tail") or ""
    if "out of memory" in stderr_text.lower() or "OutOfMemoryError" in stderr_text:
        record["status"] = "oom"
    return record


def classify(record: dict) -> str:
    """Say whether this batch size is limited by the reader or by the GPU."""
    reader = record.get("avg_reader_cost") or 0.0
    batch_cost = record.get("avg_batch_cost") or 0.0
    if batch_cost <= 0:
        return "unknown"
    ratio = reader / batch_cost
    if ratio > 0.6:
        return "reader-bound (raise --workers)"
    if (record.get("peak_gpu_reserved_mb") or 0) > 0.85 * TOTAL_GPU_MB:
        return "near VRAM limit"
    return "gpu-bound"


TOTAL_GPU_MB = 6141


def main() -> int:
    args = parse_args()

    if args.workers < 0:
        args.workers = max(2, min(8, (os.cpu_count() or 4) - 2))

    try:
        import subprocess as sp
        query = sp.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20)
        if query.returncode == 0 and query.stdout.strip():
            TOTAL_GPU_MB = int(query.stdout.strip().splitlines()[0])
    except Exception:
        pass

    sizes = []
    for chunk in args.sizes.split(","):
        chunk = chunk.strip()
        if chunk:
            # Round down to a multiple of 16 so MultiScaleSampler stays valid.
            sizes.append(max(16, int(int(chunk) // 16) * 16))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    print(f"probing batch sizes {sizes} | workers {args.workers} | subset {args.subset} | "
          f"GPU {TOTAL_GPU_MB} MB")
    print("-" * 78)

    results = []
    for attempt in range(max(1, args.repeat)):
        for batch in sizes:
            label = f"batch {batch:>4}" + (f" (pass {attempt + 1})" if args.repeat > 1 else "") + " ... "
            print(label, end="", flush=True)
            record = run_candidate(batch, args.workers, args)
            record["pass"] = attempt + 1
            results.append(record)
            if record.get("steady_state_ips"):
                note = classify(record)
                print(f"{record['steady_state_ips']:>7.1f} img/s  "
                      f"peak {record.get('peak_gpu_reserved_mb')} MB  "
                      f"({note})  [{record['status']}]")
            else:
                print(f"no throughput recorded  [{record['status']}]")
            if args.cooldown > 0 and not (attempt == args.repeat - 1 and batch == sizes[-1]):
                time.sleep(args.cooldown)

    # A size is judged on its best pass: throttling only ever makes a run look slower.
    best_per_size: dict[int, dict] = {}
    for record in results:
        if not record.get("steady_state_ips") or record["status"] == "oom":
            continue
        current = best_per_size.get(record["batch_size"])
        if current is None or record["steady_state_ips"] > current["steady_state_ips"]:
            best_per_size[record["batch_size"]] = record

    usable = list(best_per_size.values())
    best = max(usable, key=lambda r: r["steady_state_ips"], default=None)

    payload = {
        "generated_at": reports.now(),
        "gpu_memory_mb": TOTAL_GPU_MB,
        "workers": args.workers,
        "subset": args.subset,
        "amp_level": args.amp_level,
        "shared_memory": args.shared_memory,
        "candidates": results,
        "best_per_size": {str(size): record for size, record in sorted(best_per_size.items())},
        "recommended_batch_size": best["batch_size"] if best else None,
        "recommended_ips": best["steady_state_ips"] if best else None,
        "bottleneck": classify(best) if best else "unknown",
    }
    destination = args.report_dir / "batch_probe.json"
    reports.write_json(destination, payload)

    print("-" * 78)
    if best:
        print(f"recommended batch size: {best['batch_size']} at {best['steady_state_ips']:.1f} img/s "
              f"({payload['bottleneck']})")
    else:
        print("no candidate completed; check PaddleOCR/output/probe_b*/train.log")
    print(f"report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
