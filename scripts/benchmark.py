"""Benchmark one pipeline phase: training throughput, memory, and accuracy.

Combines the parsed PaddleOCR training log with the phase's evaluation report so every
phase can be compared on the same axes, and writes both a JSON record and a combined
markdown/HTML table across phases.

    python scripts/benchmark.py --phase smoke
    python scripts/benchmark.py --phase full --compare-to reports/val_base_arabic_ppocrv5.json
    python scripts/benchmark.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import logparse
    from lib import reports
    from lib.paths import OUTPUT_ROOT, PHASES, REPORT_ROOT, phase_export_dir, phase_output_dir
else:  # pragma: no cover
    from .lib import logparse
    from .lib import reports
    from .lib.paths import OUTPUT_ROOT, PHASES, REPORT_ROOT, phase_export_dir, phase_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=PHASES, help="Phase to benchmark")
    parser.add_argument("--train-only", action="store_true", help="Skip the accuracy section")
    parser.add_argument("--compare-to", type=Path, help="Baseline evaluation JSON (e.g. the base model report)")
    parser.add_argument("--summary", action="store_true", help="Print a comparison table across phases and exit")
    parser.add_argument("--report-dir", type=Path, default=REPORT_ROOT)
    return parser.parse_args()


def find_evaluation(phase: str, report_dir: Path, split: str) -> Path | None:
    for candidate in (f"{split}_kurdish_{phase}.json", f"val_kurdish_{phase}.json", f"test_kurdish_{phase}.json"):
        path = report_dir / candidate
        if path.is_file():
            return path
    return None


def evaluate_exported(phase: str, split: str, report_dir: Path) -> dict | None:
    """Score the phase's exported model, exporting it first if that has not happened yet."""
    from evaluate import main as evaluate_main  # imported lazily: pulls in paddle

    model_dir = phase_export_dir(phase)
    if not (model_dir / "inference.pdiparams").is_file():
        return None
    argv = sys.argv
    sys.argv = ["evaluate.py", "--model-dir", str(model_dir), "--split", split,
                "--name", f"kurdish_{phase}", "--no-csv", "--output-dir", str(report_dir), "--progress-every", "0"]
    try:
        evaluate_main()
    finally:
        sys.argv = argv
    return json.loads((report_dir / f"{split}_kurdish_{phase}.json").read_text(encoding="utf-8"))


def build_payload(phase: str, args: argparse.Namespace) -> dict:
    output_dir = phase_output_dir(phase)
    parsed = logparse.parse_log(output_dir / "train.log")
    training = logparse.summarise(parsed)

    payload: dict = {
        "phase": phase,
        "generated_at": reports.now(),
        "output_dir": str(output_dir),
        "log_present": not parsed.get("missing", True),
        "training": training,
        "evaluation": {},
    }

    if args.train_only:
        return payload

    comparison = None
    if args.compare_to and args.compare_to.is_file():
        comparison = args.compare_to.read_text(encoding="utf-8")

    evaluation_payload = find_evaluation(phase, args.report_dir, "val")
    if evaluation_payload is None:
        evaluation_payload = find_evaluation(phase, args.report_dir, "test")
    if evaluation_payload is not None:
        payload["evaluation"] = json.loads(evaluation_payload.read_text(encoding="utf-8"))
        payload["evaluation_source"] = str(evaluation_payload)

    if comparison:
        from lib import metrics as mx
        baseline = json.loads(comparison)
        if payload["evaluation"]:
            payload["improvement_vs_baseline"] = mx.compare(baseline, payload["evaluation"])
        payload["baseline"] = str(args.compare_to)

    checkpoints = sorted(
        path.name for path in output_dir.glob("*.pdparams")
    ) if output_dir.is_dir() else []
    payload["checkpoints"] = checkpoints
    payload["has_best_accuracy"] = "best_accuracy.pdparams" in checkpoints
    return payload


def render_table(payloads: list[dict]) -> str:
    lines = [
        "| Phase | Epochs | img/s | Peak GPU MB | CER | WER | Exact | vs base |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for payload in payloads:
        training = payload.get("training") or {}
        evaluation = (payload.get("evaluation") or {}).get("overall", {})
        improvement = payload.get("improvement_vs_baseline") or {}
        cer_verdict = ""
        if "cer" in improvement:
            row = improvement["cer"]
            cer_verdict = f"{row['verdict']} ({row['delta'] * 100:+.2f}pp)"
        lines.append(
            f"| {payload['phase']} "
            f"| {training.get('epochs_completed', '-')} "
            f"| {training.get('steady_state_ips') or '-'} "
            f"| {training.get('peak_gpu_reserved_mb') or '-'} "
            f"| {evaluation.get('cer', '-') if evaluation else '-'} "
            f"| {evaluation.get('wer', '-') if evaluation else '-'} "
            f"| {evaluation.get('exact_match_accuracy', '-') if evaluation else '-'} "
            f"| {cer_verdict or '-'} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.summary:
        payloads = []
        for phase in PHASES:
            path = args.report_dir / f"benchmark_{phase}.json"
            if path.is_file():
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
        if not payloads:
            print("no benchmark records found; run a phase first", file=sys.stderr)
            return 1
        table = render_table(payloads)
        print(table)
        (args.report_dir / "benchmark_summary.md").write_text(
            "# Pipeline benchmark summary\n\n" + table + "\n", encoding="utf-8"
        )
        return 0

    if not args.phase:
        print("error: pass --phase or --summary", file=sys.stderr)
        return 2

    payload = build_payload(args.phase, args)
    destination = args.report_dir / f"benchmark_{args.phase}.json"
    reports.write_json(destination, payload)

    training = payload["training"]
    print(f"\nbenchmark: {args.phase}")
    print("-" * 60)
    if not payload["log_present"]:
        print("  no training log found; run scripts\\train.ps1 -Phase " + args.phase)
    else:
        print(f"  epochs completed   {training.get('epochs_completed')}")
        print(f"  steady-state speed {training.get('steady_state_ips') or 0:.1f} img/s")
        print(f"  peak GPU reserved  {training.get('peak_gpu_reserved_mb')} MB")
        print(f"  peak GPU allocated {training.get('peak_gpu_allocated_mb')} MB")
        print(f"  avg reader cost    {training.get('avg_reader_cost', 0):.4f} s")
        print(f"  last train loss    {training.get('last_loss')}")
    evaluation = (payload.get("evaluation") or {}).get("overall")
    if evaluation:
        print(f"  CER                {evaluation['cer']:.4%}")
        print(f"  WER                {evaluation['wer']:.4%}")
        print(f"  exact match        {evaluation['exact_match_accuracy']:.4%}")
    for key, row in (payload.get("improvement_vs_baseline") or {}).items():
        print(f"    {key:22s} {row['baseline']:.4%} -> {row['candidate']:.4%}  ({row['verdict']})")
    print(f"\n  report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
