"""Production training orchestrator and multi-checkpoint evaluation manager."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.benchmark.engine import BenchmarkEngine
from src.utils.config_injector import PaddleConfigInjector
from src.utils.logger import setup_logger
from src.utils.visualizer import Visualizer

logger = setup_logger("Trainer")


class PaddleTrainer:
    """End-to-end trainer managing PaddleOCR execution, monitoring, and auto-benchmarking."""

    def __init__(self, paddleocr_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize trainer instance.

        Args:
            paddleocr_dir: Path to PaddleOCR repository directory containing tools/train.py.
        """
        self.paddleocr_dir = self._find_paddleocr_dir(paddleocr_dir)

    @staticmethod
    def _find_paddleocr_dir(explicit_dir: Optional[Union[str, Path]] = None) -> Path:
        """Locate PaddleOCR checkout directory.

        Args:
            explicit_dir: Optional candidate directory.

        Returns:
            Resolved Path to PaddleOCR directory.

        Raises:
            FileNotFoundError: If PaddleOCR repo cannot be located.
        """
        candidates = [
            Path(explicit_dir) if explicit_dir else None,
            Path("./PaddleOCR"),
            Path("../PaddleOCR"),
            Path(__file__).resolve().parent.parent.parent / "PaddleOCR",
            Path(__file__).resolve().parent.parent.parent.parent / "PaddleOCR",
        ]
        for c in candidates:
            if c and (c / "tools" / "train.py").is_file():
                return c.resolve()

        raise FileNotFoundError("Could not locate PaddleOCR repository containing tools/train.py")

    def run_training(
        self,
        config_path: Union[str, Path],
        val_label_path: Union[str, Path],
        dict_path: Union[str, Path],
        save_model_dir: Union[str, Path],
        gpus: str = "0",
        auto_benchmark: bool = True,
    ) -> list[dict[str, Any]]:
        """Launch training subprocess, log output in real-time, and benchmark checkpoints.

        Args:
            config_path: Path to serialized runtime YAML config.
            val_label_path: Path to val_label.txt for post-training validation.
            dict_path: Path to vocabulary dictionary.txt.
            save_model_dir: Directory where checkpoints are saved.
            gpus: CUDA GPU identifiers (e.g. "0" or "0,1").
            auto_benchmark: Whether to run benchmark leaderboard on finished checkpoints.

        Returns:
            List of benchmark record dictionaries for all evaluated checkpoints.
        """
        config_p = Path(config_path).resolve()
        val_label_p = Path(val_label_path).resolve()
        dict_p = Path(dict_path).resolve()
        save_dir_p = Path(save_model_dir).resolve()

        train_py = self.paddleocr_dir / "tools" / "train.py"
        is_multi_gpu = "," in gpus and len(gpus.split(",")) > 1

        if is_multi_gpu:
            cmd = [
                sys.executable,
                "-m",
                "paddle.distributed.launch",
                f"--gpus={gpus}",
                str(train_py),
                "-c",
                str(config_p),
            ]
        else:
            cmd = [
                sys.executable,
                str(train_py),
                "-c",
                str(config_p),
            ]

        logger.info(f"Initiating training process with command: {' '.join(cmd)}")
        import paddle
        env = os.environ.copy()
        has_cuda = bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
        if has_cuda and gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus
        env["PYTHONPATH"] = f"{str(self.paddleocr_dir)}{os.pathsep}{env.get('PYTHONPATH', '')}"

        process = subprocess.Popen(
            cmd,
            cwd=str(self.paddleocr_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )

        recent_lines: list[str] = []
        if process.stdout:
            for line in process.stdout:
                line_str = line.strip()
                recent_lines.append(line_str)
                if len(recent_lines) > 50:
                    recent_lines.pop(0)

                lower = line_str.lower()
                if "epoch:" in lower or "loss:" in lower or "acc:" in lower or "speed:" in lower:
                    logger.info(f"[TRAIN] {line_str}")
                elif "save model in" in lower or "best model is saved" in lower or "best metric" in lower:
                    logger.info(f"[CHECKPOINT] {line_str}")
                elif "error" in lower or "exception" in lower or "traceback" in lower or "fail" in lower:
                    logger.warning(f"[STDERR] {line_str}")
                elif "train dataloader has" in lower or "load pretrain successful" in lower or "backbone frozen" in lower or "backbone unfrozen" in lower:
                    logger.info(f"[INIT] {line_str}")

        process.wait()
        if process.returncode != 0:
            error_tail = "\n".join(recent_lines[-25:])
            raise RuntimeError(f"Training failed with returncode {process.returncode}:\n{error_tail}")

        logger.info(f"Training process completed. Checkpoints written to: {save_dir_p}")

        benchmark_results: list[dict[str, Any]] = []
        if auto_benchmark:
            benchmark_results = self.evaluate_all_checkpoints(
                checkpoints_dir=save_dir_p,
                config_path=config_p,
                val_label_path=val_label_p,
                dict_path=dict_p,
            )

        return benchmark_results

    def evaluate_all_checkpoints(
        self,
        checkpoints_dir: Union[str, Path],
        config_path: Union[str, Path],
        val_label_path: Union[str, Path],
        dict_path: Union[str, Path],
        max_samples: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Discover and benchmark all .pdparams checkpoints in the output directory.

        Args:
            checkpoints_dir: Directory containing saved checkpoints.
            config_path: Path to configuration YAML.
            val_label_path: Path to validation dataset labels.
            dict_path: Path to dictionary.txt.
            max_samples: Optional limit on evaluation samples.

        Returns:
            List of benchmark result dictionaries sorted by lowest CER.
        """
        checkpoints_dir = Path(checkpoints_dir).resolve()
        all_pdparams = list(checkpoints_dir.glob("*.pdparams"))
        # Exclude optimizer states and pretrain copies if present
        checkpoint_files = [f for f in all_pdparams if not f.name.endswith(".pdopt") and "pretrain" not in f.name.lower()]

        if not checkpoint_files:
            logger.warning(f"No valid .pdparams checkpoints found in {checkpoints_dir}")
            return []

        logger.info(f"Discovered {len(checkpoint_files)} checkpoints to benchmark.")

        engine = BenchmarkEngine(
            config_path=config_path,
            dict_path=dict_path,
        )

        results: list[dict[str, Any]] = []
        for ckpt in checkpoint_files:
            logger.info(f"Evaluating checkpoint: {ckpt.name}")
            report_path = checkpoints_dir / f"benchmark_{ckpt.stem}.json"
            res = engine.run_benchmark(
                model_path=ckpt,
                val_label_path=val_label_path,
                max_samples=max_samples,
                output_report_path=report_path,
            )
            record = {
                "model": ckpt.name,
                "checkpoint_path": str(ckpt),
                "accuracy": res.accuracy,
                "micro_cer": res.micro_cer,
                "micro_wer": res.micro_wer,
                "avg_latency_ms": res.avg_latency_ms,
                "throughput_fps": res.throughput_fps,
            }
            results.append(record)

        # Sort by best micro CER (lowest first)
        results.sort(key=lambda x: x["micro_cer"])

        # Display terminal leaderboard table
        table_str = Visualizer.render_table(results)
        print("\n" + table_str + "\n")

        # Generate markdown report
        report_md_path = checkpoints_dir / "benchmark_report.md"
        Visualizer.generate_markdown_report(results, report_md_path)
        logger.info(f"Markdown benchmark leaderboard generated at: {report_md_path}")

        return results
