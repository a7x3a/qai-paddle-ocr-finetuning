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


@dataclass
class TextLine:
    """Represents a recognized text segment with its spatial bounding box."""
    text: str
    score: float
    box: list[list[float]]  # 4 corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    line_number: int
    column_index: int = 0


@dataclass
class DocumentResult:
    """Complete single-page document OCR result."""
    full_text: str
    lines: list[TextLine]
    annotated_image: Optional[np.ndarray] = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    page_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "full_text": self.full_text,
            "line_count": len(self.lines),
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

        logger.info(f"Loading Text Detector from {self.det_model_p.name}...")
        args = init_args().parse_args([])
        args.det_model_dir = str(self.det_model_p)
        args.use_gpu = use_gpu
        args.det_limit_side_len = det_limit_side_len
        self.detector = predict_det.TextDetector(args)

        logger.info(f"Loading Kurdish Recognizer from {self.rec_model_p.name}...")
        self.recognizer = Recognizer(self.rec_model_p, use_gpu=use_gpu)

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
    def _sort_boxes_rtl(cls, dt_boxes: list[np.ndarray], img_width: int = 1000) -> list[tuple[np.ndarray, int]]:
        """Sort detected bounding boxes in natural Kurdish/Arabic reading order:
        Column-aware (Right column first, then Left column), Top-to-bottom, and Right-to-Left within each line.
        """
        if len(dt_boxes) <= 1:
            return [(b, 0) for b in dt_boxes]

        columns = cls._detect_columns(dt_boxes, img_width)
        final_ordered: list[tuple[np.ndarray, int]] = []

        for col_idx, col_boxes in columns:
            boxes = list(col_boxes)
            boxes.sort(key=lambda b: (b[:, 1].min(), -b[:, 0].max()))

            lines: list[list[np.ndarray]] = []
            for box in boxes:
                box_top = float(box[:, 1].min())
                box_bottom = float(box[:, 1].max())
                box_height = max(1.0, box_bottom - box_top)

                placed = False
                for line in lines:
                    line_tops = [b[:, 1].min() for b in line]
                    line_bottoms = [b[:, 1].max() for b in line]
                    line_avg_top = sum(line_tops) / len(line_tops)
                    line_avg_bottom = sum(line_bottoms) / len(line_bottoms)
                    line_avg_height = max(1.0, line_avg_bottom - line_avg_top)

                    v_center = (box_top + box_bottom) / 2.0
                    if line_avg_top - (line_avg_height * 0.35) <= v_center <= line_avg_bottom + (line_avg_height * 0.35):
                        line.append(box)
                        placed = True
                        break

                if not placed:
                    lines.append([box])

            # Sort lines top-to-bottom
            lines.sort(key=lambda l: min(b[:, 1].min() for b in l))

            # Within each line, sort Kurdish/Arabic text RIGHT to LEFT (largest x_max first)
            for line in lines:
                line.sort(key=lambda b: -b[:, 0].max())
                for b in line:
                    final_ordered.append((b, col_idx))

        return final_ordered

    def read_document(
        self,
        image_input: Union[str, Path, np.ndarray],
        min_score: float = 0.3,
        annotate: bool = True,
        page_number: int = 1,
    ) -> DocumentResult:
        """Run detection and recognition on an entire document image page."""
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
                # Try recognizing whole image directly
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
                    line = TextLine(
                        text=preds[0].text.strip(),
                        score=round(float(preds[0].score), 4),
                        box=[[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]],
                        line_number=1,
                    )
                    return DocumentResult(
                        full_text=line.text,
                        lines=[line],
                        annotated_image=img if annotate else None,
                        timing_ms={"det_ms": round(det_ms, 1), "rec_ms": round(rec_ms, 1), "total_ms": round(total_ms, 1)},
                        page_number=page_number,
                    )

            total_ms = (time.perf_counter() - t_start) * 1000.0
            return DocumentResult(
                full_text="",
                lines=[],
                annotated_image=img if annotate else None,
                timing_ms={"det_ms": round(det_ms, 1), "rec_ms": 0.0, "total_ms": round(total_ms, 1)},
                page_number=page_number,
            )

        # Step 2: RTL Reading Order & Column-Aware Sorting
        sorted_pairs = self._sort_boxes_rtl(dt_boxes, img_width=img_w)
        sorted_boxes = [p[0] for p in sorted_pairs]
        col_indices = [p[1] for p in sorted_pairs]

        # Step 3: Perspective Crop of each text box
        crops: list[np.ndarray] = []
        for box in sorted_boxes:
            crop = get_rotate_crop_image(img, box)
            crops.append(crop)

        # Step 4: Batch Recognition
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

        # Step 5: Assemble recognized lines
        lines: list[TextLine] = []
        line_texts: list[str] = []
        annotated = img.copy() if annotate else None

        line_counter = 1
        for i, (box, col_idx, p) in enumerate(zip(sorted_boxes, col_indices, preds)):
            score = float(p.score)
            text = p.text.strip()
            if not text or score < min_score:
                continue

            pts = box.astype(np.int32).reshape((-1, 1, 2))
            lines.append(TextLine(
                text=text,
                score=round(score, 4),
                box=box.tolist(),
                line_number=line_counter,
                column_index=col_idx,
            ))
            line_texts.append(text)

            if annotated is not None:
                # Color code: cyan for single/col0, green for col1
                box_color = (212, 182, 6) if col_idx == 0 else (6, 212, 182)
                cv2.polylines(annotated, [pts], isClosed=True, color=box_color, thickness=2)
                x_min, y_min = int(box[:, 0].min()), int(box[:, 1].min())
                cv2.circle(annotated, (x_min, y_min), 10, (6, 182, 212), -1)
                cv2.putText(
                    annotated,
                    str(line_counter),
                    (x_min - 4, y_min + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            line_counter += 1

        full_text = "\n".join(line_texts)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return DocumentResult(
            full_text=full_text,
            lines=lines,
            annotated_image=annotated,
            timing_ms={
                "det_ms": round(det_ms, 1),
                "rec_ms": round(rec_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            page_number=page_number,
        )

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
