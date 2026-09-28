"""Shared paths and constants for the Kurdish OCR pipeline (re-exported from src.utils.paths)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.paths import (
    ASSETS_DIR,
    BASE_INFERENCE_SHA256,
    BASE_INFERENCE_URL,
    BASE_MODEL_DIR,
    BASE_PRETRAINED,
    BASE_PRETRAINED_URL,
    CONFIG_DIR,
    DATA_ROOT,
    DATASET_REPO,
    DATASET_REVISION,
    DICT_PATH,
    EXPORT_ROOT,
    MAX_TEXT_LENGTH,
    OUTPUT_ROOT,
    PADDLE_COMMIT,
    PADDLE_REPO,
    PADDLE_ROOT,
    PHASES,
    PRETRAIN_DIR,
    PRETRAINED_SHA256,
    PROJECT_ROOT,
    RAW_ROOT,
    REC_ROOT,
    REPORT_ROOT,
    SPLITS,
    TRAIN_CONFIG,
    phase_export_dir,
    phase_output_dir,
)

__all__ = [
    "PROJECT_ROOT",
    "PADDLE_ROOT",
    "PADDLE_COMMIT",
    "PADDLE_REPO",
    "DATA_ROOT",
    "RAW_ROOT",
    "REC_ROOT",
    "DATASET_REPO",
    "DATASET_REVISION",
    "SPLITS",
    "MAX_TEXT_LENGTH",
    "CONFIG_DIR",
    "TRAIN_CONFIG",
    "DICT_PATH",
    "ASSETS_DIR",
    "BASE_MODEL_DIR",
    "PRETRAIN_DIR",
    "BASE_PRETRAINED",
    "BASE_PRETRAINED_URL",
    "BASE_INFERENCE_URL",
    "PRETRAINED_SHA256",
    "BASE_INFERENCE_SHA256",
    "OUTPUT_ROOT",
    "EXPORT_ROOT",
    "REPORT_ROOT",
    "PHASES",
    "phase_output_dir",
    "phase_export_dir",
]
