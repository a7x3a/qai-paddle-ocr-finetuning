"""Parse PaddleOCR training logs into structured records.

PaddleOCR already prints everything a benchmark needs on one line per report:

    epoch: [3/40], global_step: 900, batch_cost: 0.012, lr: 0.00050, acc: 0.72,
    norm_edit_dis: 0.88, CTCLoss: 0.41, NRTRLoss: 0.33, avg_reader_cost: 0.001,
    avg_batch_cost: 0.012, avg_samples: 32, ips: 268.31 samples/s, eta: 0:12:03,
    max_mem_reserved: 5123 MB, max_mem_allocated: 4011 MB

plus `cur metric, acc: ..., norm_edit_dis: ...` and `best metric, ...` for evaluation.
Reading those avoids reimplementing instrumentation and keeps the numbers identical to
what PaddleOCR reported during the run.
"""

from __future__ import annotations

import re
from pathlib import Path

TRAIN_LINE = re.compile(
    r"epoch:\s*\[(?P<epoch>\d+)/(?P<epochs>\d+)\],\s*global_step:\s*(?P<step>\d+),\s*(?P<body>.*?)"
    r"avg_reader_cost:\s*(?P<reader>[\d.]+)\s*s,\s*"
    r"avg_batch_cost:\s*(?P<batch_cost>[\d.]+)\s*s,\s*"
    r"avg_samples:\s*(?P<samples>[\d.]+),\s*"
    r"ips:\s*(?P<ips>[\d.]+)\s*samples/s"
)
METRIC_LINE = re.compile(r"^(?P<kind>cur metric|best metric),\s*(?P<body>.+)$", re.IGNORECASE)
EVAL_IMAGE_LINE = re.compile(r"eval img(?P<index>\d+)\s*:\s*(?P<body>.*)$", re.IGNORECASE)


def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_body(body: str) -> dict:
    values: dict = {}
    for chunk in body.split(","):
        if ":" not in chunk:
            continue
        key, _, raw = chunk.partition(":")
        key = key.strip()
        raw = raw.strip()
        if not key:
            continue
        number = _number(raw)
        values[key] = number if number is not None else raw
    return values


def parse_log(path: Path) -> dict:
    """Return training steps, evaluation events and any recorded best checkpoint."""
    steps: list[dict] = []
    metrics: list[dict] = []
    progress_images: list[dict] = []

    if not path.is_file():
        return {"log": str(path), "missing": True, "steps": [], "metrics": [], "progress_images": []}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRAIN_LINE.search(line)
        if match:
            record = {
                "epoch": int(match.group("epoch")),
                "total_epochs": int(match.group("epochs")),
                "global_step": int(match.group("step")),
                "avg_reader_cost": float(match.group("reader")),
                "avg_batch_cost": float(match.group("batch_cost")),
                "avg_samples": float(match.group("samples")),
                "ips": float(match.group("ips")),
            }
            record.update(_parse_body(match.group("body")))
            for key in ("max_mem_reserved", "max_mem_allocated"):
                found = re.search(rf"{key}:\s*(\d+)\s*MB", line)
                if found:
                    record[key] = int(found.group(1))
            steps.append(record)
            continue

        match = METRIC_LINE.search(line.strip())
        if match:
            metrics.append({"kind": match.group("kind").lower(), **_parse_body(match.group("body"))})
            continue

        match = EVAL_IMAGE_LINE.search(line.strip())
        if match:
            progress_images.append({"index": int(match.group("index"))})

    return {
        "log": str(path),
        "missing": False,
        "steps": steps,
        "metrics": metrics,
        "progress_images": progress_images,
        "peak_gpu_reserved_mb": max((s["max_mem_reserved"] for s in steps if "max_mem_reserved" in s), default=None),
        "peak_gpu_allocated_mb": max((s["max_mem_allocated"] for s in steps if "max_mem_allocated" in s), default=None),
        "last_ips": steps[-1]["ips"] if steps else None,
        "epochs_seen": max((s["epoch"] for s in steps), default=0),
    }


def summarise(parsed: dict) -> dict:
    """Condense a parsed log into the numbers worth putting in a benchmark report."""
    steps = parsed.get("steps") or []
    if not steps:
        return {"steps": 0, "note": "no training step lines found in the log"}

    # The last third of a run reflects steady state better than the warmup-inclusive average.
    tail = steps[max(0, len(steps) * 2 // 3):] or steps
    ips_values = [step["ips"] for step in tail if step.get("ips")]
    losses = [step["loss"] for step in tail if isinstance(step.get("loss"), float)]
    accuracies = [step["acc"] for step in tail if isinstance(step.get("acc"), float)]
    similarities = [step["norm_edit_dis"] for step in tail if isinstance(step.get("norm_edit_dis"), float)]

    best = None
    for entry in parsed.get("metrics") or []:
        if entry.get("kind") == "best metric":
            best = {key: value for key, value in entry.items() if key != "kind"}

    epochs = sorted({step["epoch"] for step in steps})
    epoch_steps: dict[int, int] = {}
    for step in steps:
        epoch_steps[step["epoch"]] = epoch_steps.get(step["epoch"], 0) + 1

    return {
        "steps": len(steps),
        "epochs_completed": max(epochs) if epochs else 0,
        "steps_per_epoch": epoch_steps,
        "steady_state_ips": sum(ips_values) / len(ips_values) if ips_values else None,
        "peak_ips": max((step["ips"] for step in steps), default=None),
        "avg_reader_cost": sum(step["avg_reader_cost"] for step in steps) / len(steps),
        "avg_batch_cost": sum(step["avg_batch_cost"] for step in steps) / len(steps),
        "peak_gpu_reserved_mb": parsed.get("peak_gpu_reserved_mb"),
        "peak_gpu_allocated_mb": parsed.get("peak_gpu_allocated_mb"),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "last_accuracy": accuracies[-1] if accuracies else None,
        "last_similarity": similarities[-1] if similarities else None,
        "best_metric": best,
    }
