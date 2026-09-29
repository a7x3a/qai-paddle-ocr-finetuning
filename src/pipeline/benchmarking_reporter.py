"""Unified Benchmarking & Multi-Stage Reporting System for PaddleOCR Fine-Tuning.

Manages versioned benchmark artifacts in:
  reports/benchmarking/v{n}/
    ├── baseline/
    │   ├── baseline_report.json
    │   └── baseline_report.md
    ├── pilot/
    │   ├── pilot_report.json
    │   ├── pilot_report.md
    │   └── pilot_runtime_config.yml
    ├── full/
    │   ├── benchmark_report.md
    │   ├── benchmark_best_accuracy.json
    │   ├── benchmark_iter_epoch_*.json
    │   └── ...
    ├── leaderboard.md
    └── summary.json
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import setup_logger

logger = setup_logger("BenchmarkingReporter")


def get_next_report_version(base_reports_dir: Union[str, Path] = "reports/benchmarking") -> int:
    """Scan existing reports/benchmarking/v*, benchmarks/v*, and outputs/v* directories and return next version integer."""
    candidates = [
        Path(base_reports_dir).resolve(),
        Path("benchmarks").resolve(),
        Path("outputs").resolve(),
        Path("output").resolve(),
        Path("export").resolve(),
    ]
    max_v = 0
    for p in candidates:
        if p.is_dir():
            for child in p.iterdir():
                if child.is_dir() and re.match(r"^v\d+$", child.name, re.IGNORECASE):
                    try:
                        v_num = int(child.name[1:])
                        if v_num > max_v:
                            max_v = v_num
                    except ValueError:
                        pass
    return max_v + 1 if max_v > 0 else 1


def resolve_report_dir(
    version: Optional[Union[int, str]] = None,
    base_reports_dir: Union[str, Path] = "reports/benchmarking",
) -> tuple[int, Path]:
    """Resolve destination version directory path across reports/benchmarking and benchmarks."""
    base_p = Path(base_reports_dir).resolve()
    base_p.mkdir(parents=True, exist_ok=True)

    if version is None:
        v_num = get_next_report_version(base_p)
    elif isinstance(version, str):
        cleaned = version.lower().lstrip("v")
        v_num = int(cleaned) if cleaned.isdigit() else 1
    else:
        v_num = int(version)

    v_dir = base_p / f"v{v_num}"
    (v_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (v_dir / "pilot").mkdir(parents=True, exist_ok=True)
    (v_dir / "full").mkdir(parents=True, exist_ok=True)
    (v_dir / "unseen").mkdir(parents=True, exist_ok=True)

    benchmarks_v = Path("benchmarks").resolve() / f"v{v_num}"
    (benchmarks_v / "baseline").mkdir(parents=True, exist_ok=True)
    (benchmarks_v / "pilot").mkdir(parents=True, exist_ok=True)
    (benchmarks_v / "full").mkdir(parents=True, exist_ok=True)
    (benchmarks_v / "unseen").mkdir(parents=True, exist_ok=True)
    return v_num, v_dir


class BenchmarkingReporter:
    """Generates structured benchmark reports and comparison leaderboards across stages."""

    def __init__(
        self,
        version: Optional[Union[int, str]] = None,
        base_reports_dir: Union[str, Path] = "reports/benchmarking",
    ) -> None:
        self.version, self.version_dir = resolve_report_dir(version, base_reports_dir)
        self.baseline_dir = self.version_dir / "baseline"
        self.pilot_dir = self.version_dir / "pilot"
        self.full_dir = self.version_dir / "full"
        self.unseen_dir = self.version_dir / "unseen"
        self.unseen_dir.mkdir(parents=True, exist_ok=True)

        self.benchmarks_v_dir = Path("benchmarks").resolve() / f"v{self.version}"
        self.benchmarks_v_dir.mkdir(parents=True, exist_ok=True)

    def _sync_benchmarks(self) -> None:
        """Mirror all reports into top-level benchmarks/v{version} and benchmarks/leaderboard.md."""
        try:
            if self.version_dir.exists() and self.version_dir.resolve() != self.benchmarks_v_dir.resolve():
                shutil.copytree(str(self.version_dir), str(self.benchmarks_v_dir), dirs_exist_ok=True)
            top_lead = Path("benchmarks/leaderboard.md").resolve()
            curr_lead = self.version_dir / "leaderboard.md"
            if curr_lead.is_file():
                top_lead.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(curr_lead), str(top_lead))
        except Exception as e:
            logger.debug(f"Benchmarks sync notice: {e}")

    def record_baseline(
        self,
        metrics: dict[str, Any],
        model_name: str = "Base Pretrained Model",
    ) -> Path:
        """Save baseline benchmark results."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        report_json = self.baseline_dir / "baseline_report.json"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        md_content = self._render_single_stage_md(
            title=f"Stage 1: Baseline Evaluation ({model_name})",
            description="Pre-training benchmark of foundation model prior to Kurdish domain adaptation.",
            metrics=metrics,
        )
        report_md = self.baseline_dir / "baseline_report.md"
        report_md.write_text(md_content, encoding="utf-8")

        logger.info(f"Baseline benchmark recorded in: {self.baseline_dir}")
        self.update_leaderboard()
        return report_json

    def record_pilot(
        self,
        metrics: dict[str, Any],
        config_path: Optional[Union[str, Path]] = None,
        train_label_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save pilot run benchmark results."""
        self.pilot_dir.mkdir(parents=True, exist_ok=True)
        report_json = self.pilot_dir / "pilot_report.json"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if config_path and Path(config_path).is_file():
            shutil.copy2(str(config_path), str(self.pilot_dir / "pilot_runtime_config.yml"))

        if train_label_path and Path(train_label_path).is_file():
            shutil.copy2(str(train_label_path), str(self.pilot_dir / "pilot_train_label.txt"))

        md_content = self._render_single_stage_md(
            title="Stage 2: Pilot Run Verification",
            description="Mini-epoch training verification ensuring gradient propagation and metric improvement.",
            metrics=metrics,
        )
        report_md = self.pilot_dir / "pilot_report.md"
        report_md.write_text(md_content, encoding="utf-8")

        logger.info(f"Pilot benchmark recorded in: {self.pilot_dir}")
        self.update_leaderboard()
        return report_json

    def record_full_checkpoints(
        self,
        checkpoints_dir: Union[str, Path],
        benchmark_results: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Copy full training checkpoint benchmark results into full/."""
        self.full_dir.mkdir(parents=True, exist_ok=True)
        ckpt_p = Path(checkpoints_dir).resolve()

        if ckpt_p.is_dir():
            for json_file in ckpt_p.glob("benchmark_*.json"):
                shutil.copy2(str(json_file), str(self.full_dir / json_file.name))
            if (ckpt_p / "benchmark_report.md").is_file():
                shutil.copy2(str(ckpt_p / "benchmark_report.md"), str(self.full_dir / "benchmark_report.md"))
            if (ckpt_p / "config.yml").is_file():
                shutil.copy2(str(ckpt_p / "config.yml"), str(self.full_dir / "config.yml"))

        if benchmark_results:
            results_file = self.full_dir / "checkpoints_summary.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Full training benchmarks recorded in: {self.full_dir}")
        self.update_leaderboard()

    def record_unseen(self, report_data: dict[str, Any], md_report: Optional[str] = None) -> tuple[Path, Path]:
        """Save unseen multilingual benchmark report and update leaderboard."""
        self.unseen_dir.mkdir(parents=True, exist_ok=True)
        json_p = self.unseen_dir / "unseen_report.json"
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        md_p = self.unseen_dir / "unseen_report.md"
        if md_report:
            md_p.write_text(md_report, encoding="utf-8")
        logger.info(f"Unseen benchmark report recorded in: {self.unseen_dir}")
        self.update_leaderboard()
        return json_p, md_p

    def update_leaderboard(self) -> tuple[Path, Path]:
        """Aggregate baseline, pilot, full training, and unseen results into a unified leaderboard."""
        baseline_data: Optional[dict[str, Any]] = None
        base_json = self.baseline_dir / "baseline_report.json"
        if base_json.is_file():
            try:
                with open(base_json, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
            except Exception:
                pass

        pilot_data: Optional[dict[str, Any]] = None
        pilot_json = self.pilot_dir / "pilot_report.json"
        if pilot_json.is_file():
            try:
                with open(pilot_json, "r", encoding="utf-8") as f:
                    pilot_data = json.load(f)
            except Exception:
                pass

        full_runs: list[dict[str, Any]] = []
        for f in sorted(self.full_dir.glob("benchmark_*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    item = json.load(fp)
                    item["_source_file"] = f.name
                    full_runs.append(item)
            except Exception:
                pass

        # Build table rows
        rows: list[dict[str, Any]] = []

        if baseline_data:
            rows.append({
                "stage": "1. Baseline (Zero-Shot)",
                "checkpoint": Path(baseline_data.get("checkpoint_path", "Base Pretrained")).name or "arabic_base",
                "accuracy": baseline_data.get("accuracy", 0.0),
                "macro_cer": baseline_data.get("macro_cer", 0.0),
                "micro_cer": baseline_data.get("micro_cer", 0.0),
                "macro_wer": baseline_data.get("macro_wer", 0.0),
                "micro_wer": baseline_data.get("micro_wer", 0.0),
                "latency_ms": baseline_data.get("avg_latency_ms", 0.0),
                "fps": baseline_data.get("throughput_fps", 0.0),
                "is_best": False,
            })

        if pilot_data:
            rows.append({
                "stage": "2. Pilot Run (Convergence)",
                "checkpoint": Path(pilot_data.get("checkpoint_path", "pilot_ckpt")).name or "pilot_best",
                "accuracy": pilot_data.get("accuracy", 0.0),
                "macro_cer": pilot_data.get("macro_cer", 0.0),
                "micro_cer": pilot_data.get("micro_cer", 0.0),
                "macro_wer": pilot_data.get("macro_wer", 0.0),
                "micro_wer": pilot_data.get("micro_wer", 0.0),
                "latency_ms": pilot_data.get("avg_latency_ms", 0.0),
                "fps": pilot_data.get("throughput_fps", 0.0),
                "is_best": False,
            })

        best_acc = 0.0
        for item in full_runs:
            acc = float(item.get("accuracy", 0.0))
            if acc > best_acc:
                best_acc = acc

        for item in full_runs:
            acc = float(item.get("accuracy", 0.0))
            name = Path(item.get("checkpoint_path", item.get("_source_file", ""))).name
            stage_name = "3. Full Fine-Tuning"
            if "best" in name.lower():
                stage_name = "3. Full Best Accuracy"
            elif "latest" in name.lower():
                stage_name = "3. Full Latest"
            elif "epoch" in name.lower():
                m = re.search(r"epoch_(\d+)", name.lower())
                ep_num = m.group(1) if m else "?"
                stage_name = f"3. Full Epoch {ep_num}"

            rows.append({
                "stage": stage_name,
                "checkpoint": name,
                "accuracy": acc,
                "macro_cer": item.get("macro_cer", 0.0),
                "micro_cer": item.get("micro_cer", 0.0),
                "macro_wer": item.get("macro_wer", 0.0),
                "micro_wer": item.get("micro_wer", 0.0),
                "latency_ms": item.get("avg_latency_ms", 0.0),
                "fps": item.get("throughput_fps", 0.0),
                "is_best": bool(acc == best_acc and acc > 0.0 and "best" in name.lower()),
            })

        # Check for unseen report
        unseen_data: Optional[dict[str, Any]] = None
        unseen_json = self.unseen_dir / "unseen_report.json"
        if unseen_json.is_file():
            try:
                with open(unseen_json, "r", encoding="utf-8") as f:
                    unseen_data = json.load(f)
            except Exception:
                pass

        # Save summary JSON
        summary_payload = {
            "version": f"v{self.version}",
            "version_dir": str(self.version_dir),
            "stages_completed": {
                "baseline": baseline_data is not None,
                "pilot": pilot_data is not None,
                "full": len(full_runs) > 0,
                "unseen": unseen_data is not None,
            },
            "best_accuracy": best_acc,
            "leaderboard": rows,
            "unseen_evaluation": unseen_data,
        }
        summary_json = self.version_dir / "summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        # Generate markdown leaderboard
        leaderboard_md = self.version_dir / "leaderboard.md"
        md_text = self._render_leaderboard_md(rows, best_acc, unseen_data)
        leaderboard_md.write_text(md_text, encoding="utf-8")

        self._sync_benchmarks()
        return leaderboard_md, summary_json

    def _render_single_stage_md(self, title: str, description: str, metrics: dict[str, Any]) -> str:
        acc = metrics.get("accuracy", 0.0) * 100.0
        m_cer = metrics.get("macro_cer", 0.0) * 100.0
        u_cer = metrics.get("micro_cer", 0.0) * 100.0
        m_wer = metrics.get("macro_wer", 0.0) * 100.0
        u_wer = metrics.get("micro_wer", 0.0) * 100.0
        lat = metrics.get("avg_latency_ms", 0.0)
        fps = metrics.get("throughput_fps", 0.0)
        samples = metrics.get("total_samples", 0)

        lines = [
            f"# {title}",
            "",
            description,
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Exact Match Accuracy** | **{acc:.2f}%** |",
            f"| **Macro CER** | {m_cer:.2f}% |",
            f"| **Micro CER** | {u_cer:.2f}% |",
            f"| **Macro WER** | {m_wer:.2f}% |",
            f"| **Micro WER** | {u_wer:.2f}% |",
            f"| **Average Latency** | {lat:.2f} ms |",
            f"| **Throughput** | {fps:.1f} FPS |",
            f"| **Total Samples Evaluated** | {samples:,} |",
            "",
            "---",
            f"*Generated by BenchmarkingReporter v{self.version}*",
        ]
        return "\n".join(lines) + "\n"

    def _render_leaderboard_md(
        self,
        rows: list[dict[str, Any]],
        best_acc: float,
        unseen_data: Optional[dict[str, Any]] = None,
    ) -> str:
        lines = [
            f"# PaddleOCR Kurdish Recognition Leaderboard (v{self.version})",
            "",
            "Comparative multi-stage progression from un-finetuned foundation model through pilot verification and full fine-tuning.",
            "",
            "| Stage | Checkpoint | Exact Match Acc | Macro CER | Micro CER | Macro WER | Latency (ms) | Throughput (FPS) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for r in rows:
            is_best = r.get("is_best", False)
            badge = " ⭐ **(BEST)**" if is_best else ""
            acc_str = f"**{r['accuracy'] * 100:.2f}%**{badge}" if is_best else f"{r['accuracy'] * 100:.2f}%"
            m_cer = f"{r['macro_cer'] * 100:.2f}%"
            u_cer = f"{r['micro_cer'] * 100:.2f}%"
            m_wer = f"{r['macro_wer'] * 100:.2f}%"
            lat = f"{r['latency_ms']:.2f}"
            fps = f"{r['fps']:.1f}"
            ckpt_name = f"`{r['checkpoint']}`"
            lines.append(f"| {r['stage']} | {ckpt_name} | {acc_str} | {m_cer} | {u_cer} | {m_wer} | {lat} | {fps} |")

        if unseen_data:
            lines.extend([
                "",
                "## 🧪 Unseen Generalization & Anti-Catastrophic Forgetting",
                "",
                f"Evaluation against completely unseen test data (`{unseen_data.get('total_samples', 0):,}` samples) across linguistic domains:",
                "",
                "| Category | Samples | Exact Match Acc | CER | WER | Avg Confidence |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            for cat, c_data in unseen_data.get("categories", {}).items():
                acc_val = c_data.get("exact_match_acc", 0.0) * 100
                cer_val = c_data.get("cer", 0.0) * 100
                wer_val = c_data.get("wer", 0.0) * 100
                conf_val = c_data.get("avg_confidence", 0.0) * 100
                lines.append(
                    f"| **{cat}** | {c_data.get('samples', 0):,} | {acc_val:.2f}% | {cer_val:.2f}% | {wer_val:.2f}% | {conf_val:.2f}% |"
                )

        lines.extend([
            "",
            "## Pipeline Stages Overview",
            "- **Stage 1 (Baseline)**: Out-of-the-box un-finetuned foundation model evaluated against the Kurdish validation set.",
            "- **Stage 2 (Pilot)**: Fast verification run on mini-batch subset ensuring loss decreases and checkpoints persist.",
            "- **Stage 3 (Full)**: Full production fine-tuning across all epochs with continual learning and learning rate scheduling.",
            "",
            "## Metric Definitions",
            "- **Exact Match Acc**: Percentage of samples where prediction matches ground truth with zero character discrepancies.",
            "- **Macro CER**: Character Error Rate averaged unweighted across each individual sample string.",
            "- **Micro CER**: Total character Levenshtein edit distance divided by total reference character count.",
            "- **Macro WER**: Word Error Rate across sequence word boundaries.",
            "- **Latency & Throughput**: Wall-clock hardware performance measured on GPU inference.",
            "",
            "---",
            f"*Generated by BenchmarkingReporter v{self.version}*",
        ])
        return "\n".join(lines) + "\n"
