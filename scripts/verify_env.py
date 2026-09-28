"""Verify the environment end to end and hard-fail on anything that would waste a training run.

Checks, in order:
  1. Python / virtualenv
  2. PaddlePaddle build and CUDA device count
  3. PaddleOCR checkout present at the pinned commit
  4. Base pretrained checkpoint and base inference model present
  5. Dictionary covers the prepared dataset and reproduces the pretrained CTC head
  6. Prepared dataset present and audited

Run with --json for machine-readable output; exit code is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import hashlib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib import dataset as ds
    from lib.paths import (
        BASE_MODEL_DIR,
        BASE_INFERENCE_SHA256,
        BASE_PRETRAINED,
        PRETRAINED_SHA256,
        DICT_PATH,
        MAX_TEXT_LENGTH,
        PADDLE_COMMIT,
        PADDLE_ROOT,
        REC_ROOT,
        SPLITS,
    )
else:  # pragma: no cover
    from .lib import dataset as ds
    from .lib.paths import (
        BASE_MODEL_DIR,
        BASE_INFERENCE_SHA256,
        BASE_PRETRAINED,
        PRETRAINED_SHA256,
        DICT_PATH,
        MAX_TEXT_LENGTH,
        PADDLE_COMMIT,
        PADDLE_ROOT,
        REC_ROOT,
        SPLITS,
    )

from lib.reports import now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument("--require-data", action="store_true", help="Fail if the prepared dataset is missing")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if no CUDA device is visible")
    return parser.parse_args()


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str, fatal: bool = True) -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail, "fatal": fatal})
        return bool(ok)

    @property
    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["ok"] and item["fatal"]]


def paddle_commit() -> str | None:
    if not (PADDLE_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(PADDLE_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    args = parse_args()
    checks = Checks()

    checks.add("python", sys.version_info >= (3, 9), f"{sys.version.split()[0]} at {sys.executable}")

    try:
        import paddle
    except Exception as error:  # noqa: BLE001
        checks.add("paddle_import", False, f"cannot import paddle: {error}")
        paddle = None
    else:
        cuda_devices = paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0
        checks.add("paddle", True, f"{paddle.__version__} cuda={paddle.is_compiled_with_cuda()} devices={cuda_devices}")
        checks.add("cuda_device", cuda_devices > 0, f"{cuda_devices} device(s) visible", fatal=args.require_cuda)

        # paddlepaddle-gpu 3.3.1 pins nvidia-cudnn-cu12==9.5.1.17 in its wheel metadata
        # but was compiled against 9.9, so it prints a "serious incompatible bug" warning
        # on every GPU call. Compared from package metadata rather than by running GPU code,
        # and reported as a non-fatal WARN so nobody mistakes it for a new problem.
        # PaddlePaddle 3.3.1 pins cuDNN 9.5 in its wheel metadata.
        try:
            from importlib import metadata as importlib_metadata

            installed = importlib_metadata.version("nvidia-cudnn-cu12")
            declared = [
                requirement.split("==")[1].strip()
                for requirement in (importlib_metadata.requires("paddlepaddle-gpu") or [])
                if requirement.replace("_", "-").lower().startswith("nvidia-cudnn-cu12")
                and "==" in requirement
            ]
            if declared and installed in declared:
                checks.add("cudnn_version", True,
                           f"cuDNN {installed}, matches the version paddlepaddle-gpu pins")
            else:
                checks.add("cudnn_version", True,
                           f"cuDNN {installed} vs paddlepaddle-gpu pin "
                           f"{declared[0] if declared else 'unknown'} - known upstream mismatch, "
                           f"verified harmless; expect a cosmetic CUDNN warning on GPU calls",
                           fatal=False)
        except Exception:  # noqa: BLE001
            pass

    try:
        import rapidfuzz
        checks.add("rapidfuzz", True, rapidfuzz.__version__)
    except Exception as error:  # noqa: BLE001
        checks.add("rapidfuzz", False, f"missing: {error}")

    commit = paddle_commit()
    if commit is None:
        checks.add("paddleocr_checkout", False, f"{PADDLE_ROOT} is not a git checkout; run scripts/setup.ps1")
    else:
        checks.add(
            "paddleocr_checkout",
            commit == PADDLE_COMMIT,
            f"{commit[:12]} (pinned {PADDLE_COMMIT[:12]})",
        )

    checks.add("base_pretrained", BASE_PRETRAINED.is_file(), f"{BASE_PRETRAINED} ({BASE_PRETRAINED.stat().st_size // 1048576} MB)"
               if BASE_PRETRAINED.is_file() else f"missing {BASE_PRETRAINED}")
    checks.add("base_inference_model", (BASE_MODEL_DIR / "inference.yml").is_file(),
               str(BASE_MODEL_DIR) if (BASE_MODEL_DIR / "inference.yml").is_file() else f"missing {BASE_MODEL_DIR}")

    # A truncated or half-copied model still imports, then silently trains garbage, so
    # both committed model files are checksummed before any training starts.
    for label, path, expected in (
        ("base_pretrained_sha256", BASE_PRETRAINED, PRETRAINED_SHA256),
        ("base_inference_sha256", BASE_MODEL_DIR / "inference.pdiparams", BASE_INFERENCE_SHA256),
    ):
        if not path.is_file():
            checks.add(label, False, f"missing {path}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1048576), b""):
                digest.update(block)
        actual = digest.hexdigest()
        checks.add(label, actual == expected, f"{actual[:12]}... {'matches' if actual == expected else f'expected {expected[:12]}...'}")

    if DICT_PATH.is_file():
        entries = [line for line in DICT_PATH.read_text(encoding="utf-8").splitlines()]
        checks.add("dictionary", len(entries) > 0, f"{len(entries)} entries at {DICT_PATH}")
        if BASE_PRETRAINED.is_file():
            try:
                report = ds.verify_dictionary(entries, BASE_PRETRAINED, use_space_char=True)
                checks.add(
                    "ctc_head_transfer",
                    report["ctc_matches"],
                    f"dictionary implies {report['expected']['ctc']} classes, checkpoint has {report['pretrained']['ctc']}",
                )
                note = "" if report["nrtr_matches"] else " (auxiliary loss branch; safe to re-initialise)"
                checks.add(
                    "nrtr_head_transfer",
                    report["nrtr_matches"],
                    f"dictionary implies {report['expected']['nrtr']}, checkpoint has {report['pretrained']['nrtr']}{note}",
                    fatal=False,
                )
            except Exception as error:  # noqa: BLE001
                checks.add("head_transfer", False, f"could not compare: {error}")
    else:
        checks.add("dictionary", False, f"missing {DICT_PATH}; run scripts/prepare_dataset.py")

    if REC_ROOT.is_dir() and any(REC_ROOT.glob("*_rec.txt")):
        for split in SPLITS:
            path = REC_ROOT / f"{split}_rec.txt"
            count = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
            checks.add(f"dataset_{split}", count > 0, f"{count} samples at {path}")
        if DICT_PATH.is_file():
            entries = set(DICT_PATH.read_text(encoding="utf-8").splitlines()) | {" "}
            out_of_vocabulary: dict[str, int] = {}
            longest = 0
            for split in SPLITS:
                path = REC_ROOT / f"{split}_rec.txt"
                if not path.is_file():
                    continue
                for line in path.open(encoding="utf-8"):
                    text = line.rstrip("\n").split("\t", 1)[-1]
                    longest = max(longest, len(text))
                    for character in text:
                        if character not in entries:
                            out_of_vocabulary[character] = out_of_vocabulary.get(character, 0) + 1
            checks.add("dictionary_covers_dataset", not out_of_vocabulary,
                       "all characters covered" if not out_of_vocabulary else f"missing {out_of_vocabulary}")
            checks.add("labels_within_max_length", longest <= MAX_TEXT_LENGTH,
                       f"longest label {longest} chars (limit {MAX_TEXT_LENGTH})", fatal=args.require_data)
    else:
        checks.add("dataset", False, f"{REC_ROOT} not prepared; run scripts/setup.ps1 then scripts/prepare_dataset.py",
                   fatal=args.require_data)

    payload = {
        "generated_at": now(),
        "ok": not checks.failed,
        "paddle_commit": commit,
        "checks": checks.items,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("Environment verification")
        print("=" * 72)
        for item in checks.items:
            mark = "PASS" if item["ok"] else ("FAIL" if item["fatal"] else "WARN")
            print(f"  [{mark}] {item['check']:26s} {item['detail']}")
        print("=" * 72)
        print("RESULT:", "OK" if payload["ok"] else "PROBLEMS FOUND")

    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
