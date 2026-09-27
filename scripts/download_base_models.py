"""Download and verify official PaddleOCR base models if missing."""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_DIR = PROJECT_ROOT / "pretrain_models"
ASSETS_DIR = PROJECT_ROOT / "assets"
BASE_MODEL_DIR = ASSETS_DIR / "base_rec_inference"

BASE_PRETRAINED_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"
BASE_INFERENCE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/arabic_PP-OCRv5_mobile_rec_infer.tar"

BASE_PRETRAINED_PATH = PRETRAIN_DIR / "arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"
PRETRAINED_SHA256 = "67bfef7d6304fc05ad5846c65e2e4e2587218a2b1de1c46989d7abc1a7a2d855"
BASE_INFERENCE_SHA256 = "4b4271fd1dd89a40b2056e1a42e58de7c0df2fdcf1a97ca9a8916d7ec45f9143"


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest_path: Path, expected_sha256: str | None = None) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    print(f"Downloading {dest_path.name} from {url} ...", flush=True)

    def _progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100.0, count * block_size * 100.0 / total_size)
            mb = (count * block_size) / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            print(f"\r  [{percent:5.1f}%] {mb:.1f} MB / {tot_mb:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=_progress)
        print()  # newline
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed: {e}") from e

    if expected_sha256:
        actual = compute_sha256(temp_path)
        if actual != expected_sha256:
            temp_path.unlink()
            raise ValueError(f"Checksum mismatch for {dest_path.name}: expected {expected_sha256}, got {actual}")

    temp_path.replace(dest_path)
    print(f"Successfully saved {dest_path.name} ({dest_path.stat().st_size / (1024 * 1024):.1f} MB)")


def ensure_base_models() -> None:
    # 1. Pretrained model weights
    if not BASE_PRETRAINED_PATH.exists():
        print(f"Base pretrained weights not found at {BASE_PRETRAINED_PATH}")
        download_file(BASE_PRETRAINED_URL, BASE_PRETRAINED_PATH, PRETRAINED_SHA256)
    else:
        print(f"Base pretrained weights found: {BASE_PRETRAINED_PATH.name}")

    # 2. Base inference model
    infer_params = BASE_MODEL_DIR / "inference.pdiparams"
    if not infer_params.exists():
        print(f"Base inference model not found at {BASE_MODEL_DIR}")
        tar_dest = ASSETS_DIR / "arabic_PP-OCRv5_mobile_rec_infer.tar"
        download_file(BASE_INFERENCE_URL, tar_dest)
        print("Extracting inference archive...")
        with tarfile.open(tar_dest, "r") as archive:
            archive.extractall(ASSETS_DIR)
        tar_dest.unlink()
        extracted_dir = ASSETS_DIR / "arabic_PP-OCRv5_mobile_rec_infer"
        if extracted_dir.exists() and not BASE_MODEL_DIR.exists():
            extracted_dir.rename(BASE_MODEL_DIR)
        print(f"Extracted base inference model to {BASE_MODEL_DIR}")
    else:
        print(f"Base inference model found: {BASE_MODEL_DIR}")


if __name__ == "__main__":
    ensure_base_models()
