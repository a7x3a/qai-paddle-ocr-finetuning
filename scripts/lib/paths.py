"""Shared paths and constants for the Kurdish OCR pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = PROJECT_ROOT / "PaddleOCR"

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "qai-ocr-v1-small"
REC_ROOT = DATA_ROOT / "kurdish_rec"

CONFIG_DIR = PROJECT_ROOT / "configs"
TRAIN_CONFIG = CONFIG_DIR / "kurdish_rec.yml"
# The filename must contain the substring "arabic": PaddleOCR's CTCLabelDecode enables
# right-to-left reversal only when `if "arabic" in character_dict_path`
# (ppocr/postprocess/rec_postprocess.py). Without it, validation accuracy during training
# and every evaluation score Arabic script in visual order and look catastrophic.
DICT_PATH = CONFIG_DIR / "arabic_kurdish_dict.txt"

ASSETS_DIR = PROJECT_ROOT / "assets"
BASE_MODEL_DIR = ASSETS_DIR / "base_rec_inference"
BASE_PRETRAINED = PROJECT_ROOT / "pretrain_models" / "arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"

# Checksums of the two committed model files, so setup.ps1 and verify_env.py can tell a
# truncated or half-copied download from a real one. Both are verified on every run.
PRETRAINED_SHA256 = "67bfef7d6304fc05ad5846c65e2e4e2587218a2b1de1c46989d7abc1a7a2d855"
BASE_INFERENCE_SHA256 = "4b4271fd1dd89a40b2056e1a42e58de7c0df2fdcf1a97ca9a8916d7ec45f9143"

OUTPUT_ROOT = PROJECT_ROOT / "output"
EXPORT_ROOT = PROJECT_ROOT / "export"
REPORT_ROOT = PROJECT_ROOT / "reports"

# PaddleOCR release this project is pinned to. setup.ps1 checks out exactly this commit.
PADDLE_COMMIT = "2661c7c0ef5c613e8f93c6e93b2e052399f0f854"
PADDLE_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"

DATASET_REPO = "a7x3a/qai-ocr-v1-small"
DATASET_REVISION = "main"

SPLITS = ("train", "val", "test")

# Maximum label length in the prepared dataset. PaddleOCR needs the CTC time steps
# (image width 320 / 8 = 40) to exceed this, which they do. audit_dataset.py
# fails the build if any label is longer, so this can never silently truncate.
MAX_TEXT_LENGTH = 32

PHASES = ("smoke", "pilot", "full")


def phase_output_dir(phase: str) -> Path:
    """Each pipeline phase gets its own directory so phases never clobber each other."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    return OUTPUT_ROOT / f"kurdish_{phase}"


def phase_export_dir(phase: str) -> Path:
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    return EXPORT_ROOT / f"kurdish_{phase}"
