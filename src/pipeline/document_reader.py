"""End-to-end full page and document OCR pipeline for Kurdish text.

Combines lightweight PP-OCRv5 Text Detection (DBNet) with the fine-tuned
Kurdish Text Recognizer (SVTR). Supports single images, multi-column pages,
PDF documents, and multi-page books with reading-order sorting (top-to-bottom,
right-to-left) and structured exports (Markdown, JSON, TXT, annotated images).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure PaddleOCR is on Python path
PADDLE_ROOT = Path(__file__).resolve().parents[2] / "PaddleOCR"
if str(PADDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLE_ROOT))

from tools.infer.utility import get_rotate_crop_image, init_args
import tools.infer.predict_det as predict_det
from scripts.lib.recognizer import Recognizer
from src.utils.logger import setup_logger

logger = setup_logger("DocumentReader")

# Optional PyMuPDF for PDF rendering
try:
    import pymupdf as fitz  # PyMuPDF modern import
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False


def detect_script_and_direction(text: str) -> tuple[str, str]:
    """Auto-detect language script and reading direction from text content.
    Returns (script_name, direction), where:
      script_name in ["kurdish", "arabic", "english", "numeric", "mixed"]
      direction in ["rtl", "ltr"]
    """
    if not text:
        return "kurdish", "rtl"

    kurdish_specific = set("ێۆڵڕژڤچپگکە")
    ar_count = 0
    ku_count = 0
    en_count = 0
    num_count = 0

    for ch in text:
        if ch in kurdish_specific:
            ku_count += 1
            ar_count += 1
        elif ("\u0600" <= ch <= "\u06ff") or ("\u0750" <= ch <= "\u077f") or ("\u08a0" <= ch <= "\u08ff") or ("\ufb50" <= ch <= "\ufeff"):
            ar_count += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            en_count += 1
        elif ch.isdigit() or ("\u0660" <= ch <= "\u0669") or ("\u06f0" <= ch <= "\u06f9"):
            num_count += 1

    total_alpha = ar_count + en_count
    if total_alpha == 0:
        if num_count > 0:
            return "numeric", "ltr"
        return "kurdish", "rtl"

    if ar_count > 0 and en_count > 0:
        direction = "rtl" if ar_count >= en_count else "ltr"
        return "mixed", direction

    if ar_count > 0:
        script = "kurdish" if ku_count > 0 else "arabic"
        return script, "rtl"

    return "english", "ltr"


@dataclass
class TextLine:
    """Represents a recognized text line with its spatial bounding box and linguistic properties."""
    text: str
    score: float
    box: list[list[float]]  # 4 corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    line_number: int
    column_index: int = 0
    script: str = "kurdish"  # "kurdish", "arabic", "english", "numeric", "mixed"
    direction: str = "rtl"   # "rtl", "ltr"
    word_count: int = 1


@dataclass
class LayoutBlock:
    """Represents a high-level semantic layout region (Title, Header, Paragraph, Table, Footer)."""
    block_id: int
    block_type: str  # "title", "header", "paragraph", "table", "footer", "caption"
    column_index: int
    bbox: list[float]  # [x_min, y_min, x_max, y_max]
    lines: list[TextLine]
    text: str
    confidence: float
    script: str = "kurdish"
    direction: str = "rtl"

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "column_index": self.column_index,
            "bbox": [round(float(c), 1) for c in self.bbox],
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "script": self.script,
            "direction": self.direction,
            "line_count": len(self.lines),
            "lines": [asdict(l) for l in self.lines],
        }


@dataclass
class DocumentResult:
    """Complete single-page document OCR result with semantic layout structure."""
    full_text: str
    lines: list[TextLine]
    markdown_text: str = ""
    blocks: list[LayoutBlock] = field(default_factory=list)
    column_count: int = 1
    annotated_image: Optional[np.ndarray] = None
    annotated_layout_image: Optional[np.ndarray] = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    page_number: int = 1
    total_pages: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "total_pages": self.total_pages,
            "column_count": self.column_count,
            "full_text": self.full_text,
            "markdown_text": self.markdown_text or self.full_text,
            "line_count": len(self.lines),
            "block_count": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
            "lines": [asdict(l) for l in self.lines],
            "timing_ms": self.timing_ms,
        }


@dataclass
class BookResult:
    """Complete multi-page book or document OCR result."""
    pages: list[DocumentResult]
    full_text: str
    total_pages: int
    total_lines: int
    avg_confidence: float
    timing_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "total_lines": self.total_lines,
            "avg_confidence": round(self.avg_confidence, 4),
            "timing_ms": self.timing_ms,
            "full_text": self.full_text,
            "pages": [p.to_dict() for p in self.pages],
        }

    def save_outputs(self, output_dir: Union[str, Path], save_visuals: bool = True) -> dict[str, Path]:
        """Save structured book outputs: book.md, book.txt, book.json, and page annotations."""
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        txt_p = out_p / "book.txt"
        txt_p.write_text(self.full_text, encoding="utf-8")

        # Markdown formatted book with page separators
        md_lines = ["# Document / Book OCR Extraction", "", f"- **Total Pages**: {self.total_pages}", f"- **Total Lines**: {self.total_lines:,}", f"- **Average Confidence**: {self.avg_confidence * 100:.2f}%", ""]
        for p in self.pages:
            md_lines.append(f"## Page {p.page_number}")
            md_lines.append("")
            md_lines.append(p.full_text if p.full_text else "*(Blank / No text detected)*")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

        md_p = out_p / "book.md"
        md_p.write_text("\n".join(md_lines), encoding="utf-8")

        # Structured JSON
        json_p = out_p / "book.json"
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # Annotated visuals
        if save_visuals:
            vis_dir = out_p / "annotated_pages"
            vis_dir.mkdir(parents=True, exist_ok=True)
            for p in self.pages:
                if p.annotated_image is not None:
                    vis_file = vis_dir / f"page_{p.page_number:04d}.jpg"
                    cv2.imwrite(str(vis_file), p.annotated_image)

        logger.info(f"Book OCR outputs saved to: {out_p}")
        return {
            "txt": txt_p,
            "md": md_p,
            "json": json_p,
        }


class DocumentReader:
    """Two-stage OCR engine: Text Detection (DBNet) + Fine-Tuned Kurdish Recognition (SVTR)."""

    def __init__(
        self,
        rec_model_dir: Union[str, Path] = "export/kurdish_final",
        det_model_dir: Union[str, Path] = "assets/base_det_inference",
        use_gpu: bool = True,
        det_limit_side_len: int = 960,
        rec_batch_size: int = 32,
        rec_image_shape: str = "3,48,320",
        max_text_length: int = 32,
    ) -> None:
        self.rec_model_p = Path(rec_model_dir).resolve()
        self.det_model_p = Path(det_model_dir).resolve()

        if not self.rec_model_p.is_dir():
            fallback = Path("assets/base_rec_inference").resolve()
            if fallback.is_dir():
                self.rec_model_p = fallback
            else:
                raise FileNotFoundError(f"Recognition model not found: {self.rec_model_p}")

        if not self.det_model_p.is_dir():
            raise FileNotFoundError(
                f"Detection model not found: {self.det_model_p}. Ensure assets/base_det_inference is present."
            )

        logger.info(f"Loading Text Detector from {self.det_model_p.name} (det_limit_side_len={det_limit_side_len})...")
        try:
            from src.utils.pir_compat import self_heal_pir_inference_model
            self_heal_pir_inference_model(self.det_model_p)
            self_heal_pir_inference_model(self.rec_model_p)
        except Exception:
            pass
        args = init_args().parse_args([])
        args.det_model_dir = str(self.det_model_p)
        args.det_limit_side_len = det_limit_side_len
        try:
            self.detector = predict_det.TextDetector(args)
        except ValueError as e:
            err = str(e)
            if "pir::DoubleAttribute" in err or "pir::FloatAttribute" in err:
                from src.utils.pir_compat import adapt_inference_json_to_runtime
                adapt_inference_json_to_runtime(self.det_model_p)
                self.detector = predict_det.TextDetector(args)
            else:
                raise

        logger.info(f"Loading Kurdish Recognizer from {self.rec_model_p.name} (batch={rec_batch_size}, shape={rec_image_shape})...")
        self.recognizer = Recognizer(
            self.rec_model_p,
            batch_size=rec_batch_size,
            use_gpu=use_gpu,
            image_shape=rec_image_shape,
            max_text_length=max_text_length,
        )

    @staticmethod
    def _detect_columns(boxes: list[np.ndarray], img_width: int) -> list[tuple[int, list[np.ndarray]]]:
        """Detect multi-column document layouts (e.g. 2-column books).
        Returns a list of (column_idx, [boxes_in_column]).
        In RTL Kurdish, Column 1 is on the Right (larger x), Column 2 on the Left (smaller x).
        """
        if len(boxes) < 6:
            return [(0, boxes)]

        # Calculate horizontal centers
        x_centers = [(b[:, 0].min() + b[:, 0].max()) / 2.0 for b in boxes]
        hist, bin_edges = np.histogram(x_centers, bins=20, range=(0, img_width))

        # Check for a valley (gutter) between 35% and 65% of the page width
        mid_start = int(len(hist) * 0.35)
        mid_end = int(len(hist) * 0.65)
        mid_region = hist[mid_start:mid_end]

        if len(mid_region) > 0 and np.min(mid_region) <= max(1, int(np.max(hist) * 0.15)):
            gutter_bin = mid_start + int(np.argmin(mid_region))
            gutter_x = (bin_edges[gutter_bin] + bin_edges[gutter_bin + 1]) / 2.0

            # Split boxes into left and right
            left_col = []
            right_col = []
            for b in boxes:
                center = (b[:, 0].min() + b[:, 0].max()) / 2.0
                if center > gutter_x:
                    right_col.append(b)
                else:
                    left_col.append(b)

            # In RTL reading (Kurdish/Arabic): Read RIGHT column first, then LEFT column!
            cols = []
            if right_col:
                cols.append((0, right_col))
            if left_col:
                cols.append((1, left_col))
            return cols

        return [(0, boxes)]

    @classmethod
    def _reconstruct_lines(
        cls,
        recognized_boxes: list[dict[str, Any]],
        img_w: int,
        img_h: int,
    ) -> list[TextLine]:
        """Cluster word-level detection boxes into unified horizontal lines with natural reading order,
        proper inter-word spacing, and language/script auto-detection.
        """
        if not recognized_boxes:
            return []

        # Step 1: Detect full-width headers/titles spanning across the top (or center)
        full_width_boxes = []
        body_boxes = []

        for b in recognized_boxes:
            box_w = float(b["box"][:, 0].max() - b["box"][:, 0].min())
            center_y = float((b["box"][:, 1].min() + b["box"][:, 1].max()) / 2.0)
            if box_w >= img_w * 0.52 and center_y < img_h * 0.40:
                full_width_boxes.append(b)
            else:
                body_boxes.append(b)

        # Step 2: Detect columns in the body boxes (e.g. 2 columns)
        column_groups = cls._detect_columns([b["box"] for b in body_boxes], img_w) if body_boxes else [(0, [])]

        # Determine dominant script across the document to set column order (RTL vs LTR)
        doc_sample_text = " ".join(b["text"] for b in recognized_boxes[:30])
        doc_script, doc_dir = detect_script_and_direction(doc_sample_text)

        segments: list[list[dict[str, Any]]] = []
        if full_width_boxes:
            segments.append(full_width_boxes)

        if len(column_groups) > 1:
            col_map = {idx: [] for idx, _ in column_groups}
            for b in body_boxes:
                assigned_col = 0
                for c_idx, c_boxes in column_groups:
                    if any(np.array_equal(b["box"], cb) for cb in c_boxes):
                        assigned_col = c_idx
                        break
                col_map[assigned_col].append(b)

            if doc_dir == "rtl":
                # In RTL: Right column (col 0) first, then Left column (col 1)
                for c_idx in sorted(col_map.keys()):
                    if col_map[c_idx]:
                        segments.append(col_map[c_idx])
            else:
                # In LTR: Left column first, then Right column
                for c_idx in sorted(col_map.keys(), reverse=True):
                    if col_map[c_idx]:
                        segments.append(col_map[c_idx])
        elif body_boxes:
            segments.append(body_boxes)

        # Step 3: Within each segment, cluster boxes into horizontal lines
        ordered_lines: list[TextLine] = []
        line_counter = 1

        for seg_idx, seg_boxes in enumerate(segments):
            seg_boxes.sort(key=lambda b: (b["box"][:, 1].min(), b["box"][:, 0].min()))

            line_clusters: list[list[dict[str, Any]]] = []
            for b in seg_boxes:
                b_top = float(b["box"][:, 1].min())
                b_bottom = float(b["box"][:, 1].max())
                b_height = max(1.0, b_bottom - b_top)
                b_center_y = (b_top + b_bottom) / 2.0

                matched_line = None
                for line in line_clusters:
                    l_tops = [float(item["box"][:, 1].min()) for item in line]
                    l_bottoms = [float(item["box"][:, 1].max()) for item in line]
                    l_avg_top = sum(l_tops) / len(l_tops)
                    l_avg_bottom = sum(l_bottoms) / len(l_bottoms)
                    l_avg_h = max(1.0, l_avg_bottom - l_avg_top)
                    l_center_y = (l_avg_top + l_avg_bottom) / 2.0

                    overlap = max(0.0, min(b_bottom, l_avg_bottom) - max(b_top, l_avg_top))
                    if overlap / min(b_height, l_avg_h) >= 0.38 or abs(b_center_y - l_center_y) <= (l_avg_h * 0.45):
                        matched_line = line
                        break

                if matched_line is not None:
                    matched_line.append(b)
                else:
                    line_clusters.append([b])

            # Sort lines strictly top to bottom
            line_clusters.sort(key=lambda lc: min(float(item["box"][:, 1].min()) for item in lc))

            # Within each line, order boxes according to detected script and JOIN WITH PROPER SPACES
            for lc in line_clusters:
                raw_combined = " ".join(item["text"] for item in lc)
                script_name, line_dir = detect_script_and_direction(raw_combined)

                if line_dir == "rtl":
                    # In RTL Kurdish/Arabic: Order from Right to Left (largest x max first)
                    lc.sort(key=lambda item: -item["box"][:, 0].max())
                else:
                    # In LTR English/Latin: Order from Left to Right (smallest x min first)
                    lc.sort(key=lambda item: item["box"][:, 0].min())

                # Reconstruct full line text with proper single spacing!
                reconstructed_text = " ".join(item["text"].strip() for item in lc if item["text"].strip())
                if not reconstructed_text:
                    continue

                all_pts = np.vstack([item["box"] for item in lc])
                x_min, y_min = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
                x_max, y_max = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
                unified_box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

                avg_conf = float(np.mean([item["score"] for item in lc]))

                ordered_lines.append(TextLine(
                    text=reconstructed_text,
                    score=round(avg_conf, 4),
                    box=unified_box,
                    line_number=line_counter,
                    column_index=seg_idx,
                    script=script_name,
                    direction=line_dir,
                    word_count=len(lc),
                ))
                line_counter += 1

        return ordered_lines

    @classmethod
    def _analyze_document_layout(
        cls,
        lines: list[TextLine],
        img_w: int,
        img_h: int,
    ) -> tuple[list[LayoutBlock], int, str]:
        """Analyze detected lines to build high-level semantic layout blocks (Title, Header, Paragraph, Table, Footer).
        Returns (blocks, column_count, markdown_text).
        """
        if not lines:
            return [], 1, ""

        col_indices = sorted(list(set(l.column_index for l in lines)))
        col_count = len(col_indices) if col_indices else 1

        line_metrics = []
        for l in lines:
            xs = [pt[0] for pt in l.box]
            ys = [pt[1] for pt in l.box]
            x_min, x_max = float(min(xs)), float(max(xs))
            y_min, y_max = float(min(ys)), float(max(ys))
            w = max(1.0, x_max - x_min)
            h = max(1.0, y_max - y_min)
            line_metrics.append({
                "line": l,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "w": w,
                "h": h,
                "center_x": (x_min + x_max) / 2.0,
                "center_y": (y_min + y_max) / 2.0,
            })

        heights = [m["h"] for m in line_metrics]
        median_h = float(np.median(heights)) if heights else 20.0

        gaps = []
        for i in range(len(line_metrics) - 1):
            curr_m, next_m = line_metrics[i], line_metrics[i + 1]
            if curr_m["line"].column_index == next_m["line"].column_index:
                gap = next_m["y_min"] - curr_m["y_max"]
                if gap > 0:
                    gaps.append(gap)
        median_gap = float(np.median(gaps)) if gaps else (median_h * 0.4)

        blocks: list[LayoutBlock] = []
        current_block_lines: list[TextLine] = []
        current_block_type: str = "paragraph"
        current_col: int = line_metrics[0]["line"].column_index

        def flush_block():
            nonlocal current_block_lines, current_block_type, current_col
            if not current_block_lines:
                return
            b_xs = [pt[0] for l in current_block_lines for pt in l.box]
            b_ys = [pt[1] for l in current_block_lines for pt in l.box]
            bbox = [min(b_xs), min(b_ys), max(b_xs), max(b_ys)]

            if current_block_type == "paragraph":
                block_text = " ".join(l.text for l in current_block_lines)
            else:
                block_text = "\n".join(l.text for l in current_block_lines)

            avg_conf = float(np.mean([l.score for l in current_block_lines]))
            b_script, b_dir = detect_script_and_direction(block_text)

            blocks.append(LayoutBlock(
                block_id=len(blocks) + 1,
                block_type=current_block_type,
                column_index=current_col,
                bbox=bbox,
                lines=list(current_block_lines),
                text=block_text,
                confidence=avg_conf,
                script=b_script,
                direction=b_dir,
            ))
            current_block_lines = []

        for i, m in enumerate(line_metrics):
            l = m["line"]
            line_h = m["h"]
            line_w = m["w"]
            line_ymin = m["y_min"]
            line_ymax = m["y_max"]
            is_col_change = (l.column_index != current_col)

            # Heuristics for semantic layout categorization
            is_footer = (line_ymin > img_h * 0.90 and len(l.text.split()) <= 6)
            is_title = (line_h >= median_h * 1.45 or (i == 0 and line_h >= median_h * 1.25 and line_w > img_w * 0.4))
            is_header = (not is_title and (line_h >= median_h * 1.18 or (line_w < (img_w / max(1, col_count)) * 0.55 and len(l.text.split()) <= 6 and i > 0 and (line_ymin - line_metrics[i-1]["y_max"]) > median_gap * 1.5)))

            # Tabular alignment detection (multiple cells in same horizontal line)
            is_table_cell = False
            if i + 1 < len(line_metrics):
                nxt = line_metrics[i + 1]
                overlap_y = max(0.0, min(line_ymax, nxt["y_max"]) - max(line_ymin, nxt["y_min"]))
                if overlap_y / max(1.0, min(line_h, nxt["h"])) > 0.5 and abs(nxt["center_x"] - m["center_x"]) > line_w * 0.4:
                    is_table_cell = True
            if i > 0:
                prv = line_metrics[i - 1]
                overlap_y = max(0.0, min(line_ymax, prv["y_max"]) - max(line_ymin, prv["y_min"]))
                if overlap_y / max(1.0, min(line_h, prv["h"])) > 0.5 and abs(prv["center_x"] - m["center_x"]) > line_w * 0.4:
                    is_table_cell = True

            target_type = "paragraph"
            if is_title:
                target_type = "title"
            elif is_header:
                target_type = "header"
            elif is_footer:
                target_type = "footer"
            elif is_table_cell:
                target_type = "table"

            gap_from_prev = 0.0
            if current_block_lines and not is_col_change:
                gap_from_prev = line_ymin - line_metrics[i - 1]["y_max"]

            start_new = (
                is_col_change
                or (target_type != current_block_type)
                or (target_type in {"title", "header", "footer"})
                or (current_block_type in {"title", "header", "footer"})
                or (current_block_type == "paragraph" and gap_from_prev > max(median_gap * 2.2, median_h * 1.6))
            )

            if start_new and current_block_lines:
                flush_block()
                current_block_type = target_type
                current_col = l.column_index
            elif not current_block_lines:
                current_block_type = target_type
                current_col = l.column_index

            current_block_lines.append(l)

        flush_block()

        # Build clean Markdown text representation
        md_parts: list[str] = []
        for b in blocks:
            if b.block_type == "title":
                md_parts.append(f"# {b.text}\n")
            elif b.block_type == "header":
                md_parts.append(f"## {b.text}\n")
            elif b.block_type == "table":
                table_lines = [f"| {l.text} |" for l in b.lines]
                md_parts.append("\n".join(table_lines) + "\n")
            elif b.block_type == "footer":
                md_parts.append(f"*{b.text}*\n")
            else:
                md_parts.append(f"{b.text}\n")

        markdown_text = "\n".join(md_parts).strip()
        return blocks, col_count, markdown_text

    @classmethod
    def _draw_layout_annotations(
        cls,
        img: np.ndarray,
        blocks: list[LayoutBlock],
        column_count: int = 1,
    ) -> np.ndarray:
        """Render high-resolution visual layout masks, bounding boxes, and semantic badges."""
        canvas = img.copy()
        overlay = img.copy()
        img_h, img_w = img.shape[:2]

        color_map = {
            "title": (180, 50, 180),      # Deep Violet / Magenta
            "header": (220, 130, 20),     # Vibrant Blue
            "paragraph": (60, 180, 75),   # Emerald / Forest Green
            "table": (20, 140, 240),      # Amber / Orange
            "footer": (140, 140, 140),    # Slate Gray
            "caption": (190, 150, 40),    # Cyan / Teal
        }

        # Step 1: Draw translucent filled region for each block
        for b in blocks:
            color = color_map.get(b.block_type, (60, 180, 75))
            x1, y1, x2, y2 = [int(c) for c in b.bbox]
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_w - 1, x2 + pad)
            y2 = min(img_h - 1, y2 + pad)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

        # Step 2: Draw borders and semantic badges
        for b in blocks:
            color = color_map.get(b.block_type, (60, 180, 75))
            x1, y1, x2, y2 = [int(c) for c in b.bbox]
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_w - 1, x2 + pad)
            y2 = min(img_h - 1, y2 + pad)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            script_tag = "KU" if b.script == "kurdish" else ("EN" if b.script == "english" else ("AR" if b.script == "arabic" else "NUM"))
            if b.block_type == "title":
                badge_str = f"H1 TITLE [{script_tag}] #{b.block_id}"
            elif b.block_type == "header":
                badge_str = f"H2 HEADER [{script_tag}] #{b.block_id}"
            elif b.block_type == "table":
                badge_str = f"TABLE #{b.block_id} ({len(b.lines)} rows)"
            elif b.block_type == "footer":
                badge_str = f"FOOTER #{b.block_id}"
            else:
                badge_str = f"P #{b.block_id} [{script_tag}] ({len(b.lines)} lines)"

            font_scale = 0.42
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            bx1 = min(img_w - tw - 12, max(0, x2 - tw - 8))
            by1 = max(0, y1 - th - 8)
            bx2 = bx1 + tw + 8
            by2 = by1 + th + 6

            if by1 < 4:
                by1 = y1 + 2
                by2 = by1 + th + 6

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 1)
            cv2.putText(
                canvas,
                badge_str,
                (bx1 + 4, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

        return canvas

    @classmethod
    def _draw_line_annotations(
        cls,
        img: np.ndarray,
        lines: list[TextLine],
    ) -> np.ndarray:
        """Render precise line-level detection polygons, sequence IDs, and directional cues."""
        canvas = img.copy()
        for line in lines:
            pts = np.array(line.box, dtype=np.int32).reshape((-1, 1, 2))
            # Color based on script
            if line.script == "kurdish":
                box_color = (212, 182, 6)    # Cyan
            elif line.script == "english":
                box_color = (20, 140, 240)    # Amber
            elif line.script == "arabic":
                box_color = (180, 50, 180)   # Violet
            else:
                box_color = (60, 180, 75)    # Emerald

            cv2.polylines(canvas, [pts], isClosed=True, color=box_color, thickness=2)

            xs = [p[0] for p in line.box]
            ys = [p[1] for p in line.box]

            # Badge placed on the side where reading begins:
            # Right side for RTL, Left side for LTR
            if line.direction == "rtl":
                badge_x = int(max(xs))
                badge_y = int(min(ys))
            else:
                badge_x = int(min(xs))
                badge_y = int(min(ys))

            cv2.circle(canvas, (badge_x, badge_y), 9, box_color, -1)
            num_str = str(line.line_number)
            cv2.putText(
                canvas,
                num_str,
                (badge_x - (4 if len(num_str) == 1 else 6), badge_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def read_document(
        self,
        image_input: Union[str, Path, np.ndarray],
        min_score: float = 0.3,
        annotate: bool = True,
        page_number: int = 1,
    ) -> DocumentResult:
        """Run detection, smart line reconstruction with language auto-detection and proper spacing, and semantic layout analysis."""
        if isinstance(image_input, (str, Path)):
            img_p = Path(image_input).resolve()
            if not img_p.is_file():
                raise FileNotFoundError(f"Image not found: {img_p}")
            img = cv2.imread(str(img_p))
            if img is None:
                raise ValueError(f"Failed to decode image: {img_p}")
        else:
            img = image_input

        t_start = time.perf_counter()
        img_h, img_w = img.shape[:2]

        # Step 1: Text Detection
        t_det_0 = time.perf_counter()
        dt_boxes, _ = self.detector(img)
        det_ms = (time.perf_counter() - t_det_0) * 1000.0

        # Fallback: if detection finds 0 boxes, check if this is a cropped text line
        if dt_boxes is None or len(dt_boxes) == 0:
            if img_h < 150 or (img_w / max(1, img_h) > 2.0):
                t_rec_0 = time.perf_counter()
                tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tf.name, img)
                tf.close()
                try:
                    preds, _ = self.recognizer.predict_paths([Path(tf.name)])
                finally:
                    try:
                        os.unlink(tf.name)
                    except Exception:
                        pass
                rec_ms = (time.perf_counter() - t_rec_0) * 1000.0
                total_ms = (time.perf_counter() - t_start) * 1000.0

                if preds and preds[0].text.strip() and float(preds[0].score) >= min_score:
                    c_script, c_dir = detect_script_and_direction(preds[0].text.strip())
                    line = TextLine(
                        text=preds[0].text.strip(),
                        score=round(float(preds[0].score), 4),
                        box=[[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]],
                        line_number=1,
                        script=c_script,
                        direction=c_dir,
                    )
                    block = LayoutBlock(
                        block_id=1,
                        block_type="paragraph",
                        column_index=0,
                        bbox=[0, 0, img_w, img_h],
                        lines=[line],
                        text=line.text,
                        confidence=line.score,
                        script=c_script,
                        direction=c_dir,
                    )
                    return DocumentResult(
                        full_text=line.text,
                        lines=[line],
                        markdown_text=line.text,
                        blocks=[block],
                        column_count=1,
                        annotated_image=img if annotate else None,
                        annotated_layout_image=img if annotate else None,
                        timing_ms={"det_ms": round(det_ms, 1), "rec_ms": round(rec_ms, 1), "total_ms": round(total_ms, 1)},
                        page_number=page_number,
                    )

            total_ms = (time.perf_counter() - t_start) * 1000.0
            return DocumentResult(
                full_text="",
                lines=[],
                markdown_text="",
                blocks=[],
                column_count=1,
                annotated_image=img if annotate else None,
                annotated_layout_image=img if annotate else None,
                timing_ms={"det_ms": round(det_ms, 1), "rec_ms": 0.0, "total_ms": round(total_ms, 1)},
                page_number=page_number,
            )

        # Step 2: Perspective Crop of all detected bounding boxes
        crops: list[np.ndarray] = []
        for box in dt_boxes:
            crop = get_rotate_crop_image(img, box)
            crops.append(crop)

        # Step 3: Batch Recognition
        t_rec_0 = time.perf_counter()
        temp_paths: list[Path] = []
        temp_files = []
        for c in crops:
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tf.name, c)
            tf.close()
            temp_files.append(tf.name)
            temp_paths.append(Path(tf.name))

        try:
            preds, _ = self.recognizer.predict_paths(temp_paths)
        finally:
            for p in temp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass

        rec_ms = (time.perf_counter() - t_rec_0) * 1000.0

        # Step 4: Filter valid recognized boxes
        recognized_boxes: list[dict[str, Any]] = []
        for box, p in zip(dt_boxes, preds):
            text = p.text.strip()
            score = float(p.score)
            if not text or score < min_score:
                continue
            recognized_boxes.append({
                "box": box,
                "text": text,
                "score": score,
            })

        # Step 5: Smart Line Reconstruction (Proper Inter-Word Spacing, Auto-Script Detection & Natural Reading Order)
        lines = self._reconstruct_lines(recognized_boxes, img_w=img_w, img_h=img_h)

        # Step 6: Semantic Layout Analysis (Blocks, Titles, Paragraphs, Tables, Columns)
        blocks, col_count, markdown_text = self._analyze_document_layout(lines, img_w=img_w, img_h=img_h)

        # Full Text with genuine lines and proper spaces
        full_text = "\n".join(l.text for l in lines)

        # Step 7: Annotations
        annotated_line = self._draw_line_annotations(img, lines) if annotate else None
        annotated_layout = self._draw_layout_annotations(img, blocks, col_count) if annotate else None

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return DocumentResult(
            full_text=full_text,
            lines=lines,
            markdown_text=markdown_text,
            blocks=blocks,
            column_count=col_count,
            annotated_image=annotated_line,
            annotated_layout_image=annotated_layout,
            timing_ms={
                "det_ms": round(det_ms, 1),
                "rec_ms": round(rec_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            page_number=page_number,
        )

    def read_pdf_page(
        self,
        pdf_input: Union[str, Path, bytes],
        page_number: int = 1,
        dpi: int = 200,
        min_score: float = 0.3,
        annotate: bool = True,
    ) -> tuple[DocumentResult, int]:
        """Process a single page of a PDF document directly without loading the entire document.
        Returns (DocumentResult, total_pages).
        """
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF ('fitz') is required for PDF reading. Install via: pip install pymupdf")

        if isinstance(pdf_input, bytes):
            doc = fitz.open(stream=pdf_input, filetype="pdf")
        else:
            pdf_p = Path(pdf_input).resolve()
            if not pdf_p.is_file():
                raise FileNotFoundError(f"PDF document not found: {pdf_p}")
            doc = fitz.open(str(pdf_p))

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError("PDF contains no pages")

        p_idx = max(0, min(page_number - 1, total_pages - 1))
        page = doc.load_page(p_idx)

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bgr = cv2.cvtColor(
            np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3),
            cv2.COLOR_RGB2BGR,
        )
        doc.close()

        res = self.read_document(
            image_input=img_bgr,
            min_score=min_score,
            annotate=annotate,
            page_number=p_idx + 1,
        )
        res.total_pages = total_pages
        return res, total_pages

    def read_pdf(
        self,
        pdf_path: Union[str, Path],
        dpi: int = 200,
        min_score: float = 0.3,
        annotate: bool = True,
        max_pages: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BookResult:
        """Process a multi-page PDF document or book."""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF ('fitz') is required for PDF reading. Install via: pip install pymupdf")

        pdf_p = Path(pdf_path).resolve()
        if not pdf_p.is_file():
            raise FileNotFoundError(f"PDF document not found: {pdf_p}")

        doc = fitz.open(str(pdf_p))
        total_doc_pages = len(doc)
        pages_to_process = min(total_doc_pages, max_pages) if max_pages else total_doc_pages

        logger.info(f"Processing PDF book: {pdf_p.name} ({pages_to_process} of {total_doc_pages} pages, {dpi} DPI)")

        page_results: list[DocumentResult] = []
        t0 = time.perf_counter()

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for p_idx in range(pages_to_process):
            page = doc.load_page(p_idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_bgr = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)

            p_res = self.read_document(
                image_input=img_bgr,
                min_score=min_score,
                annotate=annotate,
                page_number=p_idx + 1,
            )
            p_res.total_pages = total_doc_pages
            page_results.append(p_res)

            if progress_callback:
                progress_callback(p_idx + 1, pages_to_process)
            else:
                logger.info(f"  Page {p_idx + 1}/{pages_to_process}: {len(p_res.lines)} lines recognized ({p_res.timing_ms.get('total_ms', 0):.0f} ms)")

        doc.close()

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        all_lines: list[TextLine] = []
        for p in page_results:
            all_lines.extend(p.lines)

        full_text = "\n\n".join([f"--- Page {p.page_number} ---\n{p.full_text}" for p in page_results if p.full_text])
        avg_score = float(np.mean([l.score for l in all_lines])) if all_lines else 0.0

        return BookResult(
            pages=page_results,
            full_text=full_text,
            total_pages=len(page_results),
            total_lines=len(all_lines),
            avg_confidence=avg_score,
            timing_ms={
                "total_ms": round(total_time_ms, 1),
                "avg_page_ms": round(total_time_ms / max(1, len(page_results)), 1),
            },
        )

    def read_book_folder(
        self,
        folder_path: Union[str, Path],
        min_score: float = 0.3,
        annotate: bool = True,
        max_pages: Optional[int] = None,
    ) -> BookResult:
        """Process a directory of scanned book pages (PNG, JPG, TIFF)."""
        f_dir = Path(folder_path).resolve()
        if not f_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {f_dir}")

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        page_files = sorted([f for f in f_dir.iterdir() if f.suffix.lower() in exts])
        if not page_files:
            raise FileNotFoundError(f"No image files found in: {f_dir}")

        if max_pages:
            page_files = page_files[:max_pages]

        logger.info(f"Processing book folder: {f_dir.name} ({len(page_files)} pages)")
        page_results: list[DocumentResult] = []
        t0 = time.perf_counter()

        for idx, pf in enumerate(page_files):
            p_res = self.read_document(
                image_input=pf,
                min_score=min_score,
                annotate=annotate,
                page_number=idx + 1,
            )
            page_results.append(p_res)
            logger.info(f"  Page {idx + 1}/{len(page_files)} ({pf.name}): {len(p_res.lines)} lines ({p_res.timing_ms.get('total_ms', 0):.0f} ms)")

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        all_lines = [l for p in page_results for l in p.lines]
        full_text = "\n\n".join([f"--- Page {p.page_number} ({page_files[i].name}) ---\n{p.full_text}" for i, p in enumerate(page_results) if p.full_text])
        avg_score = float(np.mean([l.score for l in all_lines])) if all_lines else 0.0

        return BookResult(
            pages=page_results,
            full_text=full_text,
            total_pages=len(page_results),
            total_lines=len(all_lines),
            avg_confidence=avg_score,
            timing_ms={
                "total_ms": round(total_time_ms, 1),
                "avg_page_ms": round(total_time_ms / max(1, len(page_results)), 1),
            },
        )
