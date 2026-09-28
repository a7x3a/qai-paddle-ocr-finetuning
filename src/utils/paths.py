"""Shared paths and repository constants for the Kurdish PaddleOCR pipeline."""

from __future__ import annotations

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PaddleOCR checkout directory
PADDLE_ROOT = PROJECT_ROOT / "PaddleOCR"
PADDLE_COMMIT = "2661c7c0ef5c613e8f93c6e93b2e052399f0f854"
PADDLE_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"

# Dataset paths
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "qai-ocr-v1-small"
REC_ROOT = DATA_ROOT / "kurdish_rec"
DATASET_REPO = "a7x3a/qai-ocr-v1-small"
DATASET_REVISION = "main"
SPLITS = ("train", "val", "test")
MAX_TEXT_LENGTH = 32

# Configuration paths
CONFIG_DIR = PROJECT_ROOT / "configs"
TRAIN_CONFIG = CONFIG_DIR / "kurdish_rec.yml"
# The filename must contain the substring "arabic": PaddleOCR's CTCLabelDecode enables
# right-to-left reversal only when `if "arabic" in character_dict_path`.
DICT_PATH = CONFIG_DIR / "arabic_kurdish_dict.txt"

# Model assets and pretrained weights
ASSETS_DIR = PROJECT_ROOT / "assets"
BASE_MODEL_DIR = ASSETS_DIR / "base_rec_inference"
PRETRAIN_DIR = PROJECT_ROOT / "pretrain_models"
BASE_PRETRAINED = PRETRAIN_DIR / "arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"

BASE_PRETRAINED_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"
BASE_INFERENCE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/arabic_PP-OCRv5_mobile_rec_infer.tar"

PRETRAINED_SHA256 = "67bfef7d6304fc05ad5846c65e2e4e2587218a2b1de1c46989d7abc1a7a2d855"
BASE_INFERENCE_SHA256 = "4b4271fd1dd89a40b2056e1a42e58de7c0df2fdcf1a97ca9a8916d7ec45f9143"

# Pipeline execution and outputs
OUTPUT_ROOT = PROJECT_ROOT / "output"
EXPORT_ROOT = PROJECT_ROOT / "export"
REPORT_ROOT = PROJECT_ROOT / "reports"
PHASES = ("smoke", "pilot", "full")


def phase_output_dir(phase: str) -> Path:
    """Return isolated output directory for a specific training phase."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    return OUTPUT_ROOT / f"kurdish_{phase}"


def phase_export_dir(phase: str) -> Path:
    """Return isolated export directory for a specific training phase."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}")
    return EXPORT_ROOT / f"kurdish_{phase}"
