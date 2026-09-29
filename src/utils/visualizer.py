"""Formatting and presentation module for OCR benchmark leaderboards.

Renders terminal-ready tables with highlighting for best-performing checkpoints
and exports comprehensive Markdown comparison reports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def _safe_format_table(table_data: list[list[Any]], headers: list[str]) -> str:
    """Format table with tabulate, safely degrading to ASCII or pipe format if encoding fails."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    # Pre-clean string cells so that any character not encodable is safely handled
    safe_data: list[list[Any]] = []
    for row in table_data:
        safe_row = []
        for cell in row:
            if isinstance(cell, str):
                try:
                    cell.encode(encoding)
                    safe_row.append(cell)
                except (UnicodeEncodeError, UnicodeError):
                    safe_row.append(cell.encode(encoding, errors="replace").decode(encoding, errors="replace"))
            else:
                safe_row.append(cell)
        safe_data.append(safe_row)

    if HAS_TABULATE:
        # 1. Try fancy Unicode grid
        try:
            rendered = tabulate(safe_data, headers=headers, tablefmt="fancy_grid")
            rendered.encode(encoding)
            return rendered
        except (UnicodeEncodeError, UnicodeError):
            pass

        # 2. Fall back to standard ASCII grid
        try:
            rendered = tabulate(safe_data, headers=headers, tablefmt="grid")
            rendered.encode(encoding)
            return rendered
        except (UnicodeEncodeError, UnicodeError):
            pass

        # 3. Fall back to simple pipe formatting
        try:
            rendered = tabulate(safe_data, headers=headers, tablefmt="pipe")
            rendered.encode(encoding)
            return rendered
        except (UnicodeEncodeError, UnicodeError):
            pass

    # Built-in ASCII fallback table
    try:
        col_widths = [
            max(len(str(item)) for item in [h] + [row[idx] for row in safe_data])
            for idx, h in enumerate(headers)
        ]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        body_lines = [
            " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            for row in safe_data
        ]
        text = "\n".join([header_line, sep_line] + body_lines)
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return "\n".join(str(row) for row in [headers] + safe_data)


class Visualizer:
    """Renders formatted ASCII/Unicode benchmark tables and Markdown reports."""

    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def render_table(cls, rows: list[dict[str, Any]], use_colors: bool = True) -> str:
        """Render leaderboard comparison table.

        Args:
            rows: List of dicts with keys: 'model', 'cer', 'wer', 'accuracy', 'latency_ms', 'fps'.
            use_colors: Whether to apply ANSI green highlighting to best values.

        Returns:
            Formatted table string.
        """
        if not rows:
            return "No benchmark results to display."

        # Find best metrics
        min_cer = min((r.get("micro_cer", r.get("cer", 1.0)) for r in rows), default=1.0)
        min_wer = min((r.get("micro_wer", r.get("wer", 1.0)) for r in rows), default=1.0)
        max_acc = max((r.get("accuracy", 0.0) for r in rows), default=0.0)

        table_data = []
        for r in rows:
            cer_val = r.get("micro_cer", r.get("cer", 0.0))
            wer_val = r.get("micro_wer", r.get("wer", 0.0))
            acc_val = r.get("accuracy", 0.0)
            lat_val = r.get("avg_latency_ms", r.get("latency_ms", 0.0))
            fps_val = r.get("throughput_fps", r.get("fps", 0.0))
            model_name = str(r.get("model", r.get("checkpoint_path", "unknown")))

            # Shorten model name for display
            display_name = Path(model_name).name if "/" in model_name or "\\" in model_name else model_name

            cer_str = f"{cer_val:.4f}"
            wer_str = f"{wer_val:.4f}"
            acc_str = f"{acc_val:.4f}"

            if use_colors:
                if abs(cer_val - min_cer) < 1e-6:
                    cer_str = f"{cls.GREEN}{cls.BOLD}{cer_str}{cls.RESET}"
                if abs(wer_val - min_wer) < 1e-6:
                    wer_str = f"{cls.GREEN}{cls.BOLD}{wer_str}{cls.RESET}"
                if abs(acc_val - max_acc) < 1e-6:
                    acc_str = f"{cls.GREEN}{cls.BOLD}{acc_str}{cls.RESET}"

            table_data.append([
                display_name,
                cer_str,
                wer_str,
                acc_str,
                f"{lat_val:.2f}",
                f"{fps_val:.1f}",
            ])

        headers = ["Model / Checkpoint", "CER (Micro)", "WER (Micro)", "Accuracy", "Latency (ms)", "Throughput (FPS)"]
        return _safe_format_table(table_data, headers)

    @classmethod
    def render_predictions(cls, rows: list[dict[str, Any]]) -> str:
        """Render formatted table for sample inference predictions."""
        if not rows:
            return "No predictions to display."

        has_ground_truth = any("expected" in r for r in rows)
        if has_ground_truth:
            headers = ["Image", "Prediction", "Confidence", "Expected", "Match"]
            table_data = []
            for r in rows:
                match_str = str(r.get("match", "-"))
                if match_str == "OK":
                    match_colored = f"{cls.GREEN}{cls.BOLD}OK{cls.RESET}"
                elif match_str == "DIFF":
                    match_colored = f"\033[91m{cls.BOLD}DIFF\033[0m"
                else:
                    match_colored = match_str
                table_data.append([
                    r.get("image", ""),
                    r.get("prediction", ""),
                    r.get("confidence", "0.0000"),
                    r.get("expected", ""),
                    match_colored,
                ])
        else:
            headers = ["Image", "Prediction", "Confidence"]
            table_data = [
                [r.get("image", ""), r.get("prediction", ""), r.get("confidence", "0.0000")]
                for r in rows
            ]

        return _safe_format_table(table_data, headers)

    @classmethod
    def generate_markdown_report(
        cls,
        rows: list[dict[str, Any]],
        output_path: Union[str, Path],
        title: str = "PaddleOCR Kurdish Recognition Fine-Tuning Leaderboard",
    ) -> Path:
        """Compile and save a GitHub-flavored Markdown benchmark report.

        Args:
            rows: List of benchmark records.
            output_path: Target path for the report file.
            title: Title header for the report.

        Returns:
            Resolved Path to the created markdown report.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# {title}",
            "",
            "Comparative evaluation of un-finetuned foundation models against domain-adapted checkpoints.",
            "",
            "| Model / Checkpoint | CER (Micro) | WER (Micro) | Exact Match Acc | Avg Latency (ms) | Throughput (FPS) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        # Determine best metrics for bolding in Markdown
        min_cer = min((r.get("micro_cer", r.get("cer", 1.0)) for r in rows), default=1.0)
        min_wer = min((r.get("micro_wer", r.get("wer", 1.0)) for r in rows), default=1.0)
        max_acc = max((r.get("accuracy", 0.0) for r in rows), default=0.0)

        for r in rows:
            cer_val = r.get("micro_cer", r.get("cer", 0.0))
            wer_val = r.get("micro_wer", r.get("wer", 0.0))
            acc_val = r.get("accuracy", 0.0)
            lat_val = r.get("avg_latency_ms", r.get("latency_ms", 0.0))
            fps_val = r.get("throughput_fps", r.get("fps", 0.0))
            model_name = str(r.get("model", r.get("checkpoint_path", "unknown")))
            display_name = Path(model_name).name if "/" in model_name or "\\" in model_name else model_name

            cer_str = f"**{cer_val:.4f}**" if abs(cer_val - min_cer) < 1e-6 else f"{cer_val:.4f}"
            wer_str = f"**{wer_val:.4f}**" if abs(wer_val - min_wer) < 1e-6 else f"{wer_val:.4f}"
            acc_str = f"**{acc_val:.4f}**" if abs(acc_val - max_acc) < 1e-6 else f"{acc_val:.4f}"

            lines.append(
                f"| `{display_name}` | {cer_str} | {wer_str} | {acc_str} | {lat_val:.2f} | {fps_val:.1f} |"
            )

        lines.extend([
            "",
            "## Metric Definitions",
            "- **CER (Micro)**: Total character Levenshtein edit distance divided by total reference character count.",
            "- **WER (Micro)**: Total word Levenshtein edit distance divided by total reference word count.",
            "- **Exact Match Acc**: Proportion of model predictions exactly matching target strings.",
            "- **Latency (ms)**: Wall-clock duration per single image recognition request.",
            "- **Throughput (FPS)**: Inference throughput in frames processed per second.",
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return output_path

    @classmethod
    def render_card(cls, title: str, items: list[tuple[str, Any]], status: str = "SUCCESS") -> str:
        """Render a clean, bordered CLI card with structured key-value information."""
        if not items:
            return f"[{title}]"

        max_key_len = max((len(str(k)) for k, _ in items), default=15)
        content_widths = [len(str(k)) + len(str(v)) + 6 for k, v in items]
        total_width = max(70, max(content_widths) if content_widths else 70, len(title) + 10)

        top = f"┌{'─' * (total_width - 2)}┐"
        mid = f"├{'─' * (total_width - 2)}┤"
        bot = f"└{'─' * (total_width - 2)}┘"

        lines = [top]
        header = f" {title} "
        lines.append(f"│{cls.BOLD}{header.ljust(total_width - 2)}{cls.RESET}│")
        lines.append(mid)

        for k, v in items:
            key_str = f"  {k}".ljust(max_key_len + 4)
            val_str = str(v)
            inner = f"{key_str}: {val_str}"
            if len(inner) > total_width - 4:
                inner = inner[:total_width - 7] + "..."
            lines.append(f"│ {inner.ljust(total_width - 4)} │")

        lines.append(bot)
        return "\n".join(lines)
