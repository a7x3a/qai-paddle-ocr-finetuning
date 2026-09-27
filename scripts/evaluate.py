"""Score an exported recognition model on a prepared split and benchmark its throughput.

    python scripts/evaluate.py --model-dir assets/base_rec_inference --split test --name base

Reports corpus-level CER/WER/accuracy, breakdowns by domain, script, label length and
image size, plus inference throughput. Writes JSON, Markdown, HTML and per-sample CSV,
and can diff any run against a baseline report with --compare-to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import dataset as ds
    from lib import metrics as mx
    from lib import reports
    from lib.paths import REC_ROOT, REPORT_ROOT
    from lib.recognizer import Recognizer
else:  # pragma: no cover
    from .lib import dataset as ds
    from .lib import metrics as mx
    from .lib import reports
    from .lib.paths import REC_ROOT, REPORT_ROOT
    from .lib.recognizer import Recognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="Exported PaddleOCR model directory")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--name", default="", help="Report name (default: the model directory name)")
    parser.add_argument("--checkpoint", default="", help="Free-text label describing which checkpoint this is")
    parser.add_argument("--data-root", type=Path, default=REC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N samples")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-csv", action="store_true", help="Skip the per-sample CSV (much smaller reports)")
    parser.add_argument("--compare-to", type=Path, help="Baseline report JSON to diff against")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stem = args.name or args.model_dir.name
    root = args.data_root

    records = ds.read_records(root, args.split)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"error: no samples in split {args.split} under {root}", file=sys.stderr)
        return 2

    print(f"scoring {len(records)} {args.split} samples with {args.model_dir}", flush=True)
    recognizer = Recognizer(args.model_dir, batch_size=args.batch_size, use_gpu=not args.cpu)
    print(f"  dictionary entries={recognizer.dictionary_size} use_space_char={recognizer.use_space_char} "
          f"rtl_reverse={recognizer.reverse_applied}", flush=True)
    if not recognizer.reverse_applied:
        print(
            "error: right-to-left reversal is disabled, so Arabic-script output would be scored in "
            "visual order. This is a bug in the dictionary path, not a model problem.",
            file=sys.stderr,
        )
        recognizer.cleanup()
        return 4
    try:
        predictions, timing = recognizer.predict_paths(
            [root / record.image for record in records], progress_every=args.progress_every
        )
    finally:
        recognizer.cleanup()

    if len(predictions) != len(records):
        print("error: prediction count does not match sample count", file=sys.stderr)
        return 3

    suite = mx.MetricSuite()
    rows = []
    for record, prediction in zip(records, predictions):
        suite.add(
            record.text, prediction.text, prediction.score,
            domain=record.domain, script=record.script,
            width=record.width, height=record.height,
        )
        rows.append({
            "image": record.image,
            "domain": record.domain,
            "script": record.script,
            "ground_truth": record.text,
            "prediction": prediction.text,
            "char_errors": mx.Levenshtein.distance(record.text, prediction.text),
            "word_errors": mx.Levenshtein.distance(record.text.split(), prediction.text.split()),
            "confidence": round(prediction.score, 6),
            "exact_match": int(record.text == prediction.text),
        })

    payload = {
        "model_dir": str(args.model_dir),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "generated_at": reports.now(),
        "dictionary_entries": recognizer.dictionary_size,
        "use_space_char": recognizer.use_space_char,
        "rtl_reverse": recognizer.reverse_applied,
        "throughput": timing,
        **suite.report(),
    }

    comparison = None
    if args.compare_to:
        comparison = mx.compare(json.loads(args.compare_to.read_text(encoding="utf-8")), payload)
        payload["comparison"] = comparison

    written = reports.write_all(
        args.output_dir, f"{args.split}_{stem}", f"Kurdish OCR evaluation - {stem}",
        payload, None if args.no_csv else rows, comparison,
    )

    overall = payload["overall"]
    print("\n" + "=" * 66)
    print(f"{stem}  |  split={args.split}  |  {overall['samples']} samples")
    print("=" * 66)
    print(f"  CER                 {overall['cer']:.4%}")
    print(f"  WER                 {overall['wer']:.4%}")
    print(f"  exact match         {overall['exact_match_accuracy']:.4%}")
    print(f"  character accuracy  {overall['character_accuracy']:.4%}")
    print(f"  mean confidence     {overall['mean_confidence']:.4f}")
    print(f"  throughput          {timing['images_per_second']:.1f} img/s ({timing['seconds']:.1f}s)")
    if timing["unreadable_images"]:
        print(f"  WARNING unreadable images: {timing['unreadable_images']}")
    print("\n  by domain:")
    for name, item in payload["by_domain"].items():
        print(f"    {name:24s} n={item['samples']:>6d}  CER={item['cer']:>8.4%}  WER={item['wer']:>8.4%}  exact={item['exact_match_accuracy']:>8.4%}")
    print("\n  by script:")
    for name, item in payload["by_script"].items():
        print(f"    {name:24s} n={item['samples']:>6d}  CER={item['cer']:>8.4%}  WER={item['wer']:>8.4%}  exact={item['exact_match_accuracy']:>8.4%}")
    if comparison:
        print("\n  vs baseline:")
        for key, row in comparison.items():
            arrow = "improved" if row["verdict"] == "improved" else ("regressed" if row["verdict"] == "regressed" else "unchanged")
            print(f"    {key:24s} {row['baseline']:.4%} -> {row['candidate']:.4%}  ({arrow}, {row['delta'] * 100:+.4f}pp)")
    print("\n  reports:")
    for kind, path in written.items():
        print(f"    {kind:5s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
