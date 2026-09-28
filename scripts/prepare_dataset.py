"""Convert the QAI OCR WebDataset shards into PaddleOCR recognition data.

Produces, under --output (default data/kurdish_rec):

    images/{train,val,test}/<key>.<ext>
    {train,val,test}_rec.txt      image<TAB>transcription, UTF-8, tab separated
    {train,val,test}_meta.jsonl   per-sample domain/script/source/size, row-aligned
    dataset_summary.json          counts, character inventory, domain breakdown

and writes the training dictionary to configs/arabic_kurdish_dict.txt, preserving the
base checkpoint's character order so the pretrained recognition head transfers.
The "arabic" substring in that filename is required by PaddleOCR's RTL handling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import dataset as ds
    from lib.paths import (
        BASE_MODEL_DIR,
        CONFIG_DIR,
        DATASET_REPO,
        DICT_PATH,
        MAX_TEXT_LENGTH,
        RAW_ROOT,
        REC_ROOT,
        SPLITS,
    )
else:  # pragma: no cover
    from .lib import dataset as ds
    from .lib.paths import (
        BASE_MODEL_DIR,
        CONFIG_DIR,
        DATASET_REPO,
        DICT_PATH,
        MAX_TEXT_LENGTH,
        RAW_ROOT,
        REC_ROOT,
        SPLITS,
    )

from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_raw_dataset(input_dir: Path, repo_id: str = DATASET_REPO) -> None:
    """Ensure raw dataset shards exist locally. If missing, download from Hugging Face."""
    has_shards = any(input_dir.glob("*/*.tar")) or any(input_dir.glob("*.tar"))
    if not has_shards:
        print(f"[*] Raw dataset shards not found at {input_dir}.", flush=True)
        print(f"[*] Downloading '{repo_id}' from Hugging Face Hub (approx. 850 MB)...", flush=True)
        from huggingface_hub import snapshot_download

        input_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(input_dir),
            allow_patterns=["*.tar", "*.jsonl", "*.json", "*.md"],
        )
        print(f"[+] Hugging Face download complete: {input_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=RAW_ROOT, help="Downloaded dataset root")
    parser.add_argument("--output", type=Path, default=REC_ROOT)
    parser.add_argument("--max-text-length", type=int, default=MAX_TEXT_LENGTH)
    parser.add_argument("--limit", type=int, default=0, help="Keep only the first N samples per split (testing)")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes (default: min(cpu, 8))")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dictionary", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip if dataset is already prepared")
    return parser.parse_args()


def process_shard(job: tuple[str, str, str, int, bool]) -> dict:
    """Extract one shard. Runs in a worker process, so it takes only plain types."""
    split, shard_path, output_root, max_text_length, full_verify = job
    output_root = Path(output_root)
    image_root = output_root / "images" / split
    image_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, dict]] = []
    characters: Counter[str] = Counter()
    problems: list[str] = []
    seen: set[str] = set()

    with tarfile.open(shard_path, "r") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        keys = sorted({Path(name).stem for name in members if name.endswith(".txt")})
        for key in keys:
            if key in seen:
                problems.append(f"duplicate key {key}")
                continue
            seen.add(key)
            label_member = members.get(f"{key}.txt")
            image_member = next((members.get(key + suffix) for suffix in IMAGE_SUFFIXES if key + suffix in members), None)
            if label_member is None or image_member is None:
                problems.append(f"unpaired sample {key}")
                continue

            handle = archive.extractfile(label_member)
            if handle is None:
                problems.append(f"unreadable label {key}")
                continue
            text = handle.read().decode("utf-8").rstrip("\n").rstrip("\r")
            if not text:
                problems.append(f"empty label {key}")
                continue
            if "\n" in text or "\r" in text:
                problems.append(f"multiline label {key}")
                continue
            if len(text) > max_text_length:
                problems.append(f"label over {max_text_length} chars ({len(text)}): {key}")
                continue

            source = archive.extractfile(image_member)
            if source is None:
                problems.append(f"unreadable image {key}")
                continue
            suffix = Path(image_member.name).suffix.lower()
            relative = Path("images") / split / f"{key}{suffix}"
            destination = output_root / relative
            with destination.open("wb") as target:
                shutil.copyfileobj(source, target)

            try:
                with Image.open(destination) as image:
                    width, height = image.size
                    if full_verify:
                        image.verify()
            except Exception as error:  # noqa: BLE001 - any decode failure disqualifies the sample
                destination.unlink(missing_ok=True)
                problems.append(f"undecodable image {key}: {error}")
                continue
            if width <= 0 or height <= 0:
                destination.unlink(missing_ok=True)
                problems.append(f"empty image {key}")
                continue

            characters.update(text)
            rows.append((relative.as_posix(), text, {"image": relative.as_posix(), "width": width, "height": height}))

    return {"split": split, "shard": Path(shard_path).name, "rows": rows, "characters": dict(characters), "problems": problems}


def load_sidecar_metadata(input_root: Path, split: str) -> dict[str, dict]:
    """metadata.jsonl carries the per-sample language/domain labels we report on."""
    path = input_root / split / "metadata.jsonl"
    if not path.is_file():
        return {}
    table: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            inner = entry.get("original_metadata") or {}
            table[entry["key"]] = {
                "domain": inner.get("domain") or "unknown",
                "script": inner.get("script") or "unknown",
                "source_dataset": inner.get("source_dataset") or "unknown",
                "original_filename": entry.get("original_filename") or "",
            }
    return table


def main() -> int:
    args = parse_args()
    jobs = args.jobs or min(8, (__import__("os").cpu_count() or 4))

    # Fast skip if dataset is already prepared
    train_rec = args.output / "train_rec.txt"
    val_rec = args.output / "val_rec.txt"
    if args.skip_if_exists and train_rec.is_file() and val_rec.is_file():
        print(f"[OK] Prepared dataset already exists at {args.output}. Skipping extraction.")
        return 0

    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            if train_rec.is_file():
                print(f"[OK] Prepared dataset already exists at {args.output}. Use --overwrite to re-extract.", flush=True)
                return 0
            print(f"error: {args.output} is not empty. Re-run with --overwrite.", file=sys.stderr)
            return 2
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    # Automatically ensure raw dataset exists (download from Hugging Face if needed)
    ensure_raw_dataset(args.input)

    summary: dict = {"dataset": DATASET_REPO, "max_text_length": args.max_text_length, "splits": {}}
    all_characters: Counter[str] = Counter()
    all_problems: list[str] = []

    for split in SPLITS:
        split_root = args.input / split
        if not split_root.is_dir():
            print(f"error: missing {split_root} after downloading dataset.", file=sys.stderr)
            return 2
        shards = sorted(split_root.glob("shard-*.tar"))
        if not shards:
            print(f"error: no shard-*.tar in {split_root}", file=sys.stderr)
            return 2
        if args.limit:
            # Each shard holds ~5000 samples; only schedule enough shards to satisfy the limit.
            per_shard = 5000
            shards = shards[: max(1, -(-args.limit // per_shard) + 1)]

        sidecar = load_sidecar_metadata(args.input, split)
        tasks = [(split, str(shard), str(args.output), args.max_text_length, False) for shard in shards]
        print(f"[{split}] extracting {len(shards)} shard(s) with {jobs} worker(s)...", flush=True)

        records_tmp = args.output / f"_{split}_records.tmp"
        kept = 0
        split_characters: Counter[str] = Counter()
        with records_tmp.open("w", encoding="utf-8", newline="\n") as sink, ProcessPoolExecutor(max_workers=jobs) as pool:
            # pool.map yields in submission order, so the output stays deterministic
            # regardless of how the workers are scheduled.
            for result in pool.map(process_shard, tasks):
                for image, text, extra in result["rows"]:
                    if args.limit and kept >= args.limit:
                        break
                    key = Path(image).stem
                    extra.update(sidecar.get(key, {}))
                    extra["split"] = split
                    sink.write(f"{image}\t{text}\n")
                    sink.write(json.dumps(extra, ensure_ascii=False) + "\n")
                    split_characters.update(text)
                    all_characters.update(text)
                    kept += 1
                all_problems.extend(f"{split}/{problem}" for problem in result["problems"])
                print(f"  {result['shard']}: {len(result['rows'])} samples", flush=True)
                if args.limit and kept >= args.limit:
                    break

        if kept == 0:
            records_tmp.unlink(missing_ok=True)
            print(f"error: no usable samples in split {split}", file=sys.stderr)
            return 3

        annotation_lines: list[str] = []
        meta_lines: list[str] = []
        with records_tmp.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                (annotation_lines if index % 2 == 0 else meta_lines).append(line.rstrip("\n"))
        records_tmp.unlink()

        (args.output / f"{split}_rec.txt").write_text("\n".join(annotation_lines) + "\n", encoding="utf-8")
        (args.output / f"{split}_meta.jsonl").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

        summary["splits"][split] = {
            "samples": len(annotation_lines),
            "shards": len(shards),
            "characters": len(split_characters),
        }
        print(f"[{split}] wrote {len(annotation_lines)} samples", flush=True)

    summary["total_samples"] = sum(item["samples"] for item in summary["splits"].values())
    summary["dictionary_characters_in_dataset"] = len(all_characters)

    if not args.skip_dictionary:
        base_entries = ds.load_base_dictionary(BASE_MODEL_DIR)
        entries, appended = ds.build_dictionary(base_entries, set(all_characters), use_space_char=True)
        ds.write_dictionary(DICT_PATH, entries)
        missing_from_base = sorted(set(all_characters) - set(base_entries) - {" "})
        summary["dictionary"] = {
            "path": str(DICT_PATH),
            "base_entries": len(base_entries),
            "appended_for_dataset": appended,
            "appended_count": len(appended),
            "dataset_chars_missing_from_base_excluding_space": missing_from_base,
        }
        print(f"\ndictionary: {len(entries)} entries -> {DICT_PATH}")
        if appended:
            print(f"  appended {len(appended)} char(s) not in the base dictionary: {''.join(appended)!r}")
        else:
            print("  no new characters needed; pretrained head order preserved")

    if all_problems:
        summary["rejected_samples"] = len(all_problems)
        summary["rejected_examples"] = all_problems[:50]
        print(f"\nrejected {len(all_problems)} sample(s); first few: {all_problems[:5]}")

    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
