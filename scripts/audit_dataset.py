"""Audit the prepared recognition dataset and write a report.

Verifies, per split: annotation/label-file alignment, tab separation, image existence,
image decodability, duplicate paths, empty labels, label length against the config's
max_text_length, and dictionary coverage. Image decodability is checked on a sample by
default because fully decoding 162k images costs minutes and training would surface the
same failures anyway; pass --verify-images all to check every one.

Exit code is non-zero on any fatal problem, so the pipeline can gate on it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import dataset as ds
    from lib.paths import DICT_PATH, MAX_TEXT_LENGTH, REC_ROOT, SPLITS
    from lib.reports import now
else:  # pragma: no cover
    from .lib import dataset as ds
    from .lib.paths import DICT_PATH, MAX_TEXT_LENGTH, REC_ROOT, SPLITS
    from .lib.reports import now

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REC_ROOT)
    parser.add_argument("--max-text-length", type=int, default=MAX_TEXT_LENGTH)
    parser.add_argument("--verify-images", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--report", type=Path, default=None, help="Defaults to <root>/audit_report.json")
    return parser.parse_args()


def audit_split(
    root: Path, split: str, vocabulary: set[str], max_text_length: int, verify: str, sample_size: int
) -> dict:
    annotation_path = root / f"{split}_rec.txt"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"missing {annotation_path}")

    records = ds.read_records(root, split)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    missing_images: list[str] = []
    out_of_vocabulary: Counter[str] = Counter()
    lengths: list[int] = []
    domains: Counter[str] = Counter()
    scripts: Counter[str] = Counter()
    empty_labels = 0
    too_long = 0

    for number, record in enumerate(records, start=1):
        if not record.image or not record.text:
            errors.append(f"{split}_rec.txt:{number} empty path or label")
            continue
        if record.image in seen:
            duplicates.append(record.image)
        seen.add(record.image)
        if not record.text:
            empty_labels += 1
        length = len(record.text)
        lengths.append(length)
        if length > max_text_length:
            too_long += 1
        for character in record.text:
            if character not in vocabulary:
                out_of_vocabulary[character] += 1
        domains[record.domain] += 1
        scripts[record.script] += 1

    if duplicates:
        errors.append(f"{len(duplicates)} duplicate image path(s), e.g. {duplicates[:3]}")
    if empty_labels:
        errors.append(f"{empty_labels} empty label(s)")
    if too_long:
        errors.append(f"{too_long} label(s) longer than max_text_length={max_text_length}")
    if out_of_vocabulary:
        errors.append(f"characters outside the dictionary: {dict(out_of_vocabulary.most_common(20))}")
    if not lengths:
        errors.append("no samples found")

    # Image existence is a cheap stat() call; decodability is the expensive part.
    for record in records:
        if not (root / record.image).is_file():
            missing_images.append(record.image)
    if missing_images:
        errors.append(f"{len(missing_images)} missing image file(s), e.g. {missing_images[:3]}")

    checked, undecodable = 0, []
    if verify != "none" and records:
        if verify == "all":
            candidates = records
        else:
            candidates = random.Random(20260927).sample(records, min(sample_size, len(records)))
        for record in candidates:
            checked += 1
            try:
                with Image.open(root / record.image) as image:
                    image.verify()
            except Exception as error:  # noqa: BLE001
                undecodable.append(f"{record.image}: {error}")
        if undecodable:
            errors.append(f"{len(undecodable)} undecodable image(s), e.g. {undecodable[:3]}")

    if not any(record.domain != "unknown" for record in records):
        warnings.append("metadata sidecar missing; per-domain reporting will be unavailable")

    lengths.sort()
    count = len(lengths)
    return {
        "samples": count,
        "errors": errors,
        "warnings": warnings,
        "duplicates": len(duplicates),
        "missing_images": len(missing_images),
        "empty_labels": empty_labels,
        "labels_over_max_length": too_long,
        "out_of_vocabulary": dict(out_of_vocabulary.most_common(50)),
        "images_verified": checked,
        "images_undecodable": len(undecodable),
        "label_length": {
            "min": lengths[0] if count else 0,
            "median": lengths[count // 2] if count else 0,
            "p95": lengths[min(count - 1, int(count * 0.95))] if count else 0,
            "max": lengths[-1] if count else 0,
        },
        "by_domain": dict(domains.most_common()),
        "by_script": dict(scripts.most_common()),
    }


def main() -> int:
    args = parse_args()
    if not DICT_PATH.is_file():
        print(f"error: dictionary {DICT_PATH} is missing. Run scripts/prepare_dataset.py first.", file=sys.stderr)
        return 2

    entries = DICT_PATH.read_text(encoding="utf-8").splitlines()
    # PaddleOCR appends the space class at load time when use_space_char is true.
    vocabulary = {entry for entry in entries} | {" "}
    duplicates = sorted({entry for entry in entries if entries.count(entry) > 1}) if len(vocabulary) != len(entries) else []
    if any(len(entry) != 1 for entry in entries):
        print("error: dictionary entries must be exactly one character each", file=sys.stderr)
        return 3
    if duplicates:
        # The base checkpoint's own dictionary contains repeated characters. They are
        # dead softmax slots, but removing them would shift every class index and break
        # the pretrained head, so they are preserved deliberately and only reported.
        print(
            f"note: dictionary keeps {len(duplicates)} duplicated character(s) from the base "
            f"checkpoint ({''.join(duplicates[:12])!r}); indices are preserved on purpose",
            file=sys.stderr,
        )

    report: dict = {
        "generated_at": now(),
        "root": str(args.root),
        "dictionary": {
            "path": str(DICT_PATH),
            "entries": len(entries),
            "unique_characters": len(vocabulary) - 1,  # exclude the appended space
            "duplicated_characters": duplicates,
            "use_space_char": True,
        },
        "max_text_length": args.max_text_length,
        "verify_images": args.verify_images,
        "splits": {},
    }

    ok = True
    for split in SPLITS:
        try:
            result = audit_split(args.root, split, vocabulary, args.max_text_length, args.verify_images, args.sample_size)
        except (FileNotFoundError, ValueError) as error:
            result = {"samples": 0, "errors": [str(error)], "warnings": []}
        report["splits"][split] = result
        status = "ok" if not result["errors"] else "FAILED"
        print(f"[{status:6s}] {split:5s} {result['samples']:>7d} samples", file=sys.stderr)
        for message in result["errors"]:
            print(f"         error: {message}", file=sys.stderr)
        for message in result.get("warnings", []):
            print(f"         warn : {message}", file=sys.stderr)
        ok = ok and not result["errors"]

    report["ok"] = ok
    destination = args.report or (args.root / "audit_report.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: {key: value for key, value in result.items() if key not in ("errors", "warnings")}
                      for split, result in report["splits"].items()}, ensure_ascii=False, indent=2))
    print(f"\nreport: {destination}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
