"""Formatting and presentation module for OCR benchmark leaderboards.

Renders terminal-ready tables with highlighting for best-performing checkpoints
and exports comprehensive Markdown comparison reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


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

        if HAS_TABULATE:
            return tabulate(table_data, headers=headers, tablefmt="fancy_grid")

        # Built-in ASCII fallback table
        col_widths = [max(len(str(item)) for item in [h] + [row[idx] for row in table_data]) for idx, h in enumerate(headers)]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        body_lines = [" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) for row in table_data]
        return "\n".join([header_line, sep_line] + body_lines)

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

        if HAS_TABULATE:
            return tabulate(table_data, headers=headers, tablefmt="fancy_grid")

        col_widths = [max(len(str(item)) for item in [h] + [row[idx] for row in table_data]) for idx, h in enumerate(headers)]
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        body_lines = [" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) for row in table_data]
        return "\n".join([header_line, sep_line] + body_lines)

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
