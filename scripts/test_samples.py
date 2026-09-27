"""Run the model on individual images and print what it actually reads.

Two modes:

  Labelled samples from a prepared split, so every prediction is scored:
      python scripts/test_samples.py --model-dir export/kurdish_full --split test --count 20

  Your own images, with no ground truth:
      python scripts/test_samples.py --model-dir export/kurdish_full --images samples/*.png

Kurdish text is right-to-left. PaddleOCR only reverses it when the dictionary path
contains "arabic", so this script refuses to run if that did not happen; without the
reversal every Kurdish string comes back reversed and looks like a total failure when the
model is actually fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.metrics import MetricSuite
    from lib.paths import REC_ROOT
else:  # pragma: no cover
    from .lib.metrics import MetricSuite
    from .lib.paths import REC_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="Exported model directory")
    parser.add_argument("--split", choices=["val", "test"], help="Score labelled samples from a split")
    parser.add_argument("--images", nargs="+", type=Path, help="Your own images (no scoring)")
    parser.add_argument("--count", type=int, default=20, help="How many split samples to show")
    parser.add_argument("--offset", type=int, default=0, help="Start row in the split")
    parser.add_argument("--shuffle", action="store_true", help="Sample randomly instead of taking a prefix")
    parser.add_argument("--only-wrong", action="store_true", help="Summarise only the mismatches")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-color", action="store_true", help="Plain output, no ANSI colour")
    return parser.parse_args()


class Ink:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def dim(self, text: str) -> str:
        return self._wrap("90", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)


def collect_split_samples(split: str, offset: int, count: int, shuffle: bool) -> list[tuple[Path, str]]:
    label_file = REC_ROOT / f"{split}_rec.txt"
    if not label_file.is_file():
        raise SystemExit(f"missing {label_file}; run scripts/prepare_dataset.py first")
    rows: list[tuple[Path, str]] = []
    with label_file.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if len(rows) >= count:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((REC_ROOT / parts[0], parts[1]))
    if shuffle and len(rows) > 1:
        import random
        random.shuffle(rows)
    return rows


def collect_images(paths: list[Path]) -> list[tuple[Path, str]]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir()
                                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {path} does not exist", file=sys.stderr)
    return [(file, "") for file in files]


def main() -> int:
    args = parse_args()
    if not args.split and not args.images:
        print("error: pass --split or --images", file=sys.stderr)
        return 2

    from lib.recognizer import Recognizer

    if args.split:
        samples = collect_split_samples(args.split, args.offset, args.count, args.shuffle)
        source = f"{args.split} rows {args.offset}..{args.offset + len(samples) - 1}"
    else:
        samples = collect_images(args.images)
        source = ", ".join(str(path) for path in args.images)
    if not samples:
        print("no images to test", file=sys.stderr)
        return 1

    recognizer = Recognizer(args.model_dir, batch_size=args.batch_size, use_gpu=not args.cpu)
    if not recognizer.reverse_applied:
        print("error: right-to-left reversal is not active, so Kurdish would be read backwards.\n"
              "       The dictionary filename must contain 'arabic'.", file=sys.stderr)
        return 1

    ink = Ink(not args.no_color and sys.stdout.isatty())
    predictions, timing = recognizer.predict_paths([path for path, _ in samples])

    print()
    print(ink.bold(f"model    {args.model_dir}"))
    print(ink.bold(f"source   {source}"))
    print(ink.bold("rtl      reversal active"))
    print("-" * 78)

    suite = MetricSuite()
    exact = 0
    mismatches: list[tuple[str, str]] = []

    for (path, expected), prediction in zip(samples, predictions):
        if not prediction.readable:
            print(f"{ink.red('FAIL')} {path.name}: could not be read as an image")
            continue
        if expected:
            exact += prediction.text == expected
            suite.add(expected, prediction.text, prediction.score)
            label = ink.green("ok  ") if prediction.text == expected else ink.red("DIFF")
            if prediction.text != expected:
                mismatches.append((expected, prediction.text))
        else:
            label = ink.dim("    ")
        print(f"{label} conf {prediction.score:5.3f}  {path.name}")
        print(f"       read      {ink.dim(prediction.text)}")
        if expected:
            print(f"       expected  {ink.dim(expected)}")

    print("-" * 78)
    if expected:
        summary = suite.overall.summary()
        print(f"exact match   {exact}/{len(samples)}  ({exact / len(samples) * 100:.1f}%)")
        print(f"corpus CER    {summary['cer'] * 100:.2f}%")
        print(f"throughput    {timing['images_per_second']:.1f} img/s")
        if args.only_wrong and mismatches:
            print()
            print(ink.bold("differences"))
            for reference, hypothesis in mismatches:
                print(f"  expected {ink.dim(reference)}")
                print(f"  read     {ink.dim(hypothesis)}")
    else:
        print(f"throughput    {timing['images_per_second']:.1f} img/s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
