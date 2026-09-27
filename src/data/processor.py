"""Dataset processing, perspective warping, and vocabulary extraction pipeline.

Parses JSONL annotations (supporting both flat line-level items and nested polygon
detection boxes), executes high-precision 4-point perspective transforms, normalizes text,
generates deterministic train/val splits, and compiles the character vocabulary dictionary.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from .normalizer import KurdishTextNormalizer


@dataclass(frozen=True)
class ProcessedSample:
    """Represents a validated, processed image and transcription pair."""
    image_rel_path: str
    transcription: str


class DatasetProcessor:
    """End-to-end dataset ingestion and transformation engine."""

    def __init__(
        self,
        normalizer: Optional[KurdishTextNormalizer] = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize dataset processor with normalizer and random seed.

        Args:
            normalizer: KurdishTextNormalizer instance. If None, default instance is created.
            random_seed: Seed for deterministic train/val splitting.
        """
        self.normalizer = normalizer or KurdishTextNormalizer()
        self.random_seed = random_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    @staticmethod
    def detect_schema(record: dict[str, Any]) -> str:
        """Detect annotation schema of a JSONL record.

        Args:
            record: Decoded dictionary from JSON line.

        Returns:
            'nested_polygon' if nested annotations with coordinates exist,
            'simple_line' if direct image_path and text/transcription exist.

        Raises:
            ValueError: If the schema cannot be determined.
        """
        if "annotations" in record and isinstance(record["annotations"], list):
            return "nested_polygon"
        if ("image_path" in record or "image" in record or "filename" in record) and (
            "text" in record or "transcription" in record or "label" in record
        ):
            return "simple_line"
        raise ValueError(f"Unrecognized annotation schema in record: {list(record.keys())}")

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Order four 2D points into standard order: [top-left, top-right, bottom-right, bottom-left].

        Uses sum and difference of coordinates:
        - top-left has the smallest sum (x + y)
        - bottom-right has the largest sum (x + y)
        - top-right has the smallest difference (y - x)
        - bottom-left has the largest difference (y - x)

        Args:
            pts: NumPy array of shape (4, 2) containing (x, y) coordinates.

        Returns:
            NumPy array of shape (4, 2) in float32 ordered [TL, TR, BR, BL].
        """
        assert pts.shape == (4, 2), f"Expected shape (4, 2), got {pts.shape}"
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum coordinates: x + y
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has minimum sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has maximum sum

        # Difference coordinates: y - x
        diff = pts[:, 1] - pts[:, 0]
        rect[1] = pts[np.argmin(diff)]  # Top-right has minimum diff (x is large, y is small)
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has maximum diff (x is small, y is large)

        return rect

    @classmethod
    def four_point_transform(cls, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to crop rotated bounding polygon.

        Args:
            image: Source image array (H, W, C) or (H, W).
            pts: Array of 4 coordinate points [(x1, y1), ...].

        Returns:
            Warped, straightened image crop.
        """
        rect = cls.order_points(pts.astype(np.float32))
        (tl, tr, br, bl) = rect

        # Compute width of new image
        width_a = math.hypot(br[0] - bl[0], br[1] - bl[1])
        width_b = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
        max_width = max(int(round(max(width_a, width_b))), 8)

        # Compute height of new image
        height_a = math.hypot(tr[0] - br[0], tr[1] - br[1])
        height_b = math.hypot(tl[0] - bl[0], tl[1] - bl[1])
        max_height = max(int(round(max(height_a, height_b))), 8)

        # Construct destination points
        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    def parse_jsonl(self, jsonl_path: Path) -> Iterator[dict[str, Any]]:
        """Read and yield parsed JSON objects from a JSONL file.

        Args:
            jsonl_path: Path to the .jsonl file.

        Yields:
            Dict containing the JSON line record.
        """
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    yield json.loads(clean_line)
                except json.JSONDecodeError as err:
                    continue

    def process(
        self,
        jsonl_path: Union[str, Path],
        image_root: Union[str, Path],
        output_dir: Union[str, Path],
        val_ratio: float = 0.1,
    ) -> tuple[Path, Path, Path]:
        """Execute end-to-end dataset transformation.

        Args:
            jsonl_path: Path to raw input .jsonl file.
            image_root: Base directory containing raw images.
            output_dir: Destination directory for crops and labels.
            val_ratio: Fraction of dataset allocated to validation (e.g. 0.1).

        Returns:
            Tuple of Paths: (train_label_path, val_label_path, dictionary_path).
        """
        jsonl_path = Path(jsonl_path).resolve()
        image_root = Path(image_root).resolve()
        output_dir = Path(output_dir).resolve()
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        samples: list[ProcessedSample] = []
        character_set: set[str] = set()
        crop_counter = 0

        for record in self.parse_jsonl(jsonl_path):
            schema = self.detect_schema(record)

            if schema == "simple_line":
                img_name = record.get("image_path") or record.get("image") or record.get("filename")
                raw_text = record.get("text") or record.get("transcription") or record.get("label")
                if not img_name or not raw_text:
                    continue

                clean_text = self.normalizer.normalize(raw_text)
                if not clean_text:
                    continue

                full_img_path = image_root / img_name if not Path(img_name).is_absolute() else Path(img_name)
                if not full_img_path.is_file():
                    continue

                # Relative path from output_dir or absolute path
                rel_path = str(full_img_path.resolve())
                samples.append(ProcessedSample(image_rel_path=rel_path, transcription=clean_text))
                for char in clean_text:
                    character_set.add(char)

            elif schema == "nested_polygon":
                img_name = record.get("image_path") or record.get("image")
                if not img_name:
                    continue

                full_img_path = image_root / img_name if not Path(img_name).is_absolute() else Path(img_name)
                if not full_img_path.is_file():
                    continue

                image = cv2.imread(str(full_img_path))
                if image is None:
                    continue

                annotations = record.get("annotations", [])
                for anno in annotations:
                    raw_text = anno.get("text") or anno.get("transcription") or anno.get("label")
                    if not raw_text:
                        continue
                    clean_text = self.normalizer.normalize(raw_text)
                    if not clean_text:
                        continue

                    polygon = anno.get("polygon") or anno.get("points")
                    if not polygon or len(polygon) < 4:
                        continue

                    poly_pts = np.array(polygon[:4], dtype=np.float32)
                    try:
                        crop = self.four_point_transform(image, poly_pts)
                    except Exception:
                        continue

                    crop_counter += 1
                    crop_filename = f"crop_{crop_counter:07d}.jpg"
                    crop_path = crops_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    samples.append(
                        ProcessedSample(
                            image_rel_path=str(crop_path.resolve()),
                            transcription=clean_text,
                        )
                    )
                    for char in clean_text:
                        character_set.add(char)

        # Deterministic shuffle
        random.seed(self.random_seed)
        random.shuffle(samples)

        num_val = int(round(len(samples) * val_ratio))
        val_samples = samples[:num_val]
        train_samples = samples[num_val:]

        # Write labels
        train_label_path = output_dir / "train_label.txt"
        val_label_path = output_dir / "val_label.txt"

        with open(train_label_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(f"{s.image_rel_path}\t{s.transcription}\n")

        with open(val_label_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(f"{s.image_rel_path}\t{s.transcription}\n")

        # Write dictionary.txt (one character per line, deterministically sorted)
        dict_path = output_dir / "dictionary.txt"
        sorted_chars = sorted(list(character_set))
        with open(dict_path, "w", encoding="utf-8") as f:
            for char in sorted_chars:
                f.write(f"{char}\n")

        return train_label_path, val_label_path, dict_path

    def process_webdataset(
        self,
        input_dir: Union[str, Path] = "./data/qai-ocr-v1-small",
        output_dir: Union[str, Path] = "./data/kurdish_rec",
        max_text_length: int = 32,
        limit: int = 0,
        overwrite: bool = False,
        jobs: int = 0,
    ) -> tuple[Path, Path, Path]:
        """Process WebDataset shards (HuggingFace dataset format) into PaddleOCR recognition data.

        Extracts shard-*.tar archives, creates image directories, generates UTF-8 tab-separated
        label files ({train,val,test}_rec.txt), and verifies the character vocabulary.

        Args:
            input_dir: Path to directory containing train/, val/, test/ shard archives.
            output_dir: Destination directory for extracted images and label files.
            max_text_length: Maximum allowed label length in characters.
            limit: Optional sample count limit (0 for complete dataset).
            overwrite: Whether to overwrite existing processed data.
            jobs: Number of worker processes (0 for auto).

        Returns:
            Tuple of (train_label_path, val_label_path, dictionary_path).
        """
        import sys
        import subprocess

        in_p = Path(input_dir).resolve()
        out_p = Path(output_dir).resolve()

        train_rec = out_p / "train_rec.txt"
        val_rec = out_p / "val_rec.txt"
        dict_p = Path("configs/arabic_kurdish_dict.txt").resolve()

        # If data is already processed and overwrite is not requested, return existing paths
        if not overwrite and train_rec.is_file() and val_rec.is_file():
            train_count = sum(1 for _ in open(train_rec, encoding="utf-8"))
            val_count = sum(1 for _ in open(val_rec, encoding="utf-8"))
            if train_count > 0 and val_count > 0:
                return train_rec, val_rec, dict_p

        # Run multi-threaded WebDataset extraction
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "prepare_dataset.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--input", str(in_p),
            "--output", str(out_p),
            "--max-text-length", str(max_text_length),
        ]
        if limit > 0:
            cmd.extend(["--limit", str(limit)])
        if overwrite:
            cmd.append("--overwrite")
        if jobs > 0:
            cmd.extend(["--jobs", str(jobs)])

        subprocess.run(cmd, check=True)
        return train_rec, val_rec, dict_p
