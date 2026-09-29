"""Multilingual Unseen Dataset Evaluator for PaddleOCR.

Evaluates trained and exported models on completely unseen test data,
breaking down performance across language categories (Kurdish, Arabic, English, Numerics)
to empirically verify that the model retains 100% Arabic and English proficiency
while mastering Kurdish without catastrophic forgetting.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from scripts.lib.metrics import MetricSuite
from scripts.lib.recognizer import Recognizer
from src.utils.logger import setup_logger
from src.utils.visualizer import Visualizer

logger = setup_logger("UnseenEvaluator")

KURDISH_CHARS = set("ێۆڕڵژچپگڤ‌ھە")


def categorize_text(text: str) -> str:
    """Categorize string into linguistic domain based on character content."""
    txt = text.strip()
    has_kurdish = any(c in KURDISH_CHARS for c in txt)
    has_english = bool(re.search(r"[a-zA-Z]", txt))
    has_arabic_script = bool(re.search(r"[\u0600-\u06FF]", txt))
    is_pure_num = bool(re.match(r"^[\d\s\+\-\(\)\./#]+$", txt))

    if has_kurdish and has_english:
        return "MIXED_KURD_ENG"
    elif has_kurdish:
        return "KURDISH"
    elif has_english:
        return "ENGLISH"
    elif is_pure_num:
        return "NUMERIC"
    elif has_arabic_script:
        return "ARABIC"
    else:
        return "OTHER"


@dataclass
class CategoryMetric:
    category: str
    samples: int
    exact_match_acc: float
    cer: float
    wer: float
    avg_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "samples": self.samples,
            "exact_match_acc": round(self.exact_match_acc, 4),
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "avg_confidence": round(self.avg_confidence, 4),
        }


@dataclass
class UnseenEvaluationReport:
    model_name: str
    model_path: str
    test_file: str
    total_samples: int
    overall_accuracy: float
    overall_cer: float
    overall_wer: float
    avg_latency_ms: float
    throughput_fps: float
    categories: dict[str, CategoryMetric]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "test_file": self.test_file,
            "total_samples": self.total_samples,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "overall_cer": round(self.overall_cer, 4),
            "overall_wer": round(self.overall_wer, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "throughput_fps": round(self.throughput_fps, 2),
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "created_at": self.created_at,
        }

    def render_markdown(self) -> str:
        """Render beautiful markdown table report."""
        lines = [
            f"# 🧪 Unseen Dataset Cross-Lingual Evaluation Report",
            f"",
            f"- **Model**: `{self.model_name}` ({self.model_path})",
            f"- **Unseen Test File**: `{self.test_file}`",
            f"- **Total Samples Evaluated**: {self.total_samples:,}",
            f"- **Inference Speed**: {self.throughput_fps:.1f} img/s ({self.avg_latency_ms:.2f} ms/sample)",
            f"- **Overall Accuracy**: **{self.overall_accuracy * 100:.2f}%** | **Overall CER**: **{self.overall_cer * 100:.2f}%**",
            f"",
            f"### 🌐 Breakdown by Language / Domain",
            f"",
            f"| Category | Samples | Exact Match Acc (%) | CER (%) | WER (%) | Avg Confidence (%) |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for cat_name, m in self.categories.items():
            lines.append(
                f"| **{m.category}** | {m.samples:,} | {m.exact_match_acc * 100:.2f}% | {m.cer * 100:.2f}% | {m.wer * 100:.2f}% | {m.avg_confidence * 100:.2f}% |"
            )
        lines.append("")
        return "\n".join(lines)


class UnseenEvaluator:
    """Evaluates fine-tuned model against unseen test data with linguistic domain breakdown."""

    def __init__(
        self,
        model_dir: Union[str, Path] = "export/kurdish_final",
        batch_size: int = 32,
        use_gpu: bool = True,
        image_shape: str = "3,48,320",
        max_text_length: int = 32,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.recognizer = Recognizer(
            self.model_dir,
            batch_size=batch_size,
            use_gpu=use_gpu,
            image_shape=image_shape,
            max_text_length=max_text_length,
        )

    def evaluate(
        self,
        test_label_path: Union[str, Path] = "data/kurdish_rec/test_rec.txt",
        max_samples: Optional[int] = None,
        output_report_path: Optional[Union[str, Path]] = None,
    ) -> UnseenEvaluationReport:
        label_file = Path(test_label_path).resolve()
        if not label_file.is_file():
            raise FileNotFoundError(f"Test label file not found: {label_file}")

        rec_root = label_file.parent
        samples: list[tuple[Path, str, str]] = []

        logger.info(f"Loading unseen test samples from: {label_file}")
        all_raw: list[tuple[str, str, str]] = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    gt_text = parts[1]
                    all_raw.append((parts[0], gt_text, categorize_text(gt_text)))

        if not all_raw:
            raise ValueError(f"No valid test samples loaded from {label_file}")

        if max_samples and max_samples < len(all_raw):
            from collections import defaultdict
            by_cat: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            for item in all_raw:
                by_cat[item[2]].append(item)

            selected: list[tuple[str, str, str]] = []
            for cat_name, cat_items in by_cat.items():
                quota = max(1, int(round((len(cat_items) / len(all_raw)) * max_samples)))
                step = max(1, len(cat_items) // quota)
                selected.extend(cat_items[::step][:quota])

            samples = [(rec_root / rel, gt, cat) for rel, gt, cat in selected[:max_samples]]
        else:
            samples = [(rec_root / rel, gt, cat) for rel, gt, cat in all_raw]

        logger.info(f"Loaded {len(samples):,} unseen samples across linguistic categories.")
        img_paths = [p for p, _, _ in samples]

        start_t = time.perf_counter()
        predictions, timing = self.recognizer.predict_paths(img_paths)
        elapsed_sec = time.perf_counter() - start_t

        suites: dict[str, MetricSuite] = {
            "OVERALL": MetricSuite(),
            "KURDISH": MetricSuite(),
            "ARABIC": MetricSuite(),
            "ENGLISH": MetricSuite(),
            "NUMERIC": MetricSuite(),
            "MIXED_KURD_ENG": MetricSuite(),
        }

        for (_, gt_text, cat), pred in zip(samples, predictions):
            suites["OVERALL"].add(gt_text, pred.text, pred.score)
            if cat in suites:
                suites[cat].add(gt_text, pred.text, pred.score)
            else:
                suites.setdefault(cat, MetricSuite()).add(gt_text, pred.text, pred.score)

        categories: dict[str, CategoryMetric] = {}
        for cat_name, suite in suites.items():
            s = suite.overall.summary()
            count = s.get("samples", 0)
            if count == 0 and cat_name != "OVERALL":
                continue
            categories[cat_name] = CategoryMetric(
                category=cat_name,
                samples=count,
                exact_match_acc=s.get("exact_match_accuracy", 0.0),
                cer=s.get("cer", 0.0),
                wer=s.get("wer", 0.0),
                avg_confidence=s.get("mean_confidence", 0.0),
            )

        overall_s = suites["OVERALL"].overall.summary()
        fps = timing.get("images_per_second", len(samples) / max(elapsed_sec, 1e-4))
        avg_lat = 1000.0 / max(fps, 1e-4)

        report = UnseenEvaluationReport(
            model_name=self.model_dir.name,
            model_path=str(self.model_dir),
            test_file=str(label_file),
            total_samples=len(samples),
            overall_accuracy=overall_s.get("exact_match_accuracy", 0.0),
            overall_cer=overall_s.get("cer", 0.0),
            overall_wer=overall_s.get("wer", 0.0),
            avg_latency_ms=avg_lat,
            throughput_fps=fps,
            categories=categories,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if output_report_path:
            out_p = Path(output_report_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Unseen evaluation report saved to: {out_p}")

        return report
