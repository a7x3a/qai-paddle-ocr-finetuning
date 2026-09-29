"""Pilot training runner for rapid gradient convergence verification.

Executes a small-scale mini-epoch or 100-200 iterations on a 500-sample subset to ensure
loss stabilization and verify metric responsiveness prior to launching full multi-hour runs.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from src.benchmark.engine import BenchmarkEngine
from src.utils.config_injector import PaddleConfigInjector
from src.utils.logger import setup_logger

logger = setup_logger("PilotRun")


class PilotRunner:
    """Executes a fast pilot validation loop to observe active learning progression."""

    def __init__(self, paddleocr_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize pilot runner.

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
            Resolved Path to PaddleOCR.

        Raises:
            FileNotFoundError: If tools/train.py cannot be located.
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

    @staticmethod
    def create_subset(
        source_label: Union[str, Path],
        target_label: Union[str, Path],
        num_samples: int = 500,
        seed: int = 42,
    ) -> Path:
        """Extract a deterministic subset of samples for pilot training.

        Args:
            source_label: Original train_label.txt.
            target_label: Destination subset label file.
            num_samples: Maximum samples to include.
            seed: Random seed for sampling.

        Returns:
            Resolved Path to target_label.
        """
        source_p = Path(source_label).resolve()
        target_p = Path(target_label).resolve()
        target_p.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        with open(source_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line)

        random.seed(seed)
        random.shuffle(lines)
        subset_lines = lines[: min(num_samples, len(lines))]

        with open(target_p, "w", encoding="utf-8") as f:
            f.writelines(subset_lines)

        logger.info(f"Created pilot training subset of {len(subset_lines)} samples at: {target_p}")
        return target_p

    def run_pilot(
        self,
        template_config_path: Union[str, Path],
        character_dict_path: Union[str, Path],
        train_label_path: Union[str, Path],
        val_label_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_samples: int = 500,
        max_epochs: int = 2,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        image_shape: Optional[Union[str, list[int]]] = None,
        max_text_length: Optional[int] = 32,
        pretrained_model_path: Optional[Union[str, Path]] = None,
        max_eval_samples: Optional[int] = 200,
    ) -> dict[str, Any]:
        """Execute pilot training run and run immediate evaluation on pilot weights.

        Args:
            template_config_path: Path to baseline YAML template.
            character_dict_path: Path to dictionary.txt.
            train_label_path: Path to complete train_label.txt.
            val_label_path: Path to val_label.txt.
            output_dir: Base directory for pilot outputs.
            num_samples: Number of samples in pilot subset.
            max_epochs: Number of pilot training epochs (typically 1 or 2).
            batch_size: Batch size per iteration.
            num_workers: Number of DataLoader workers.
            image_shape: Image resolution [C,H,W].
            max_text_length: Maximum sequence length.
            pretrained_model_path: Optional path to base pretrained checkpoint.

        Returns:
            Dictionary containing pilot benchmark results and checkpoint location.
        """
        output_dir = Path(output_dir).resolve()
        pilot_dir = output_dir / "pilot_run"
        pilot_dir.mkdir(parents=True, exist_ok=True)

        if pretrained_model_path is None:
            candidates = [
                Path("pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained.pdparams"),
                Path("pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained"),
            ]
            for c in candidates:
                if c.exists() or Path(str(c) + ".pdparams").exists():
                    pretrained_model_path = c
                    break

        # 1. Create pilot dataset subset
        pilot_label = pilot_dir / "pilot_train_label.txt"
        self.create_subset(train_label_path, pilot_label, num_samples=num_samples)

        # 2. Inject pilot configuration
        injector = PaddleConfigInjector(template_config_path)
        pilot_checkpoint_dir = pilot_dir / "checkpoints"
        injector.inject_runtime_paths(
            character_dict_path=character_dict_path,
            train_label_path=pilot_label,
            val_label_path=val_label_path,
            save_model_dir=pilot_checkpoint_dir,
            pretrained_model_path=pretrained_model_path,
            batch_size=batch_size,
            epoch_num=max_epochs,
            num_workers=num_workers,
            image_shape=image_shape,
            max_text_length=max_text_length,
        )

        pilot_config_path = pilot_dir / "pilot_runtime_config.yml"
        injector.write_runtime_config(pilot_config_path)

        # 3. Launch training subprocess
        train_py = self.paddleocr_dir / "tools" / "train.py"
        cmd = [
            sys.executable,
            str(train_py),
            "-c",
            str(pilot_config_path),
            "-o",
            "Global.print_batch_step=5",
            "Global.save_epoch_step=1",
        ]

        logger.info(f"Launching pilot training run: {' '.join(cmd)}")
        env = os.environ.copy()
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
                if "loss:" in lower or "loss :" in lower:
                    logger.info(f"[PILOT TRAIN] {line_str}")
                elif "epoch:" in lower:
                    logger.info(f"[PILOT PROGRESS] {line_str}")
                elif "save model in" in lower or "best model is saved" in lower or "best metric" in lower:
                    logger.info(f"[PILOT CKPT] {line_str}")
                elif "error" in lower or "exception" in lower or "traceback" in lower:
                    logger.warning(f"[PILOT STDERR] {line_str}")

        process.wait()
        if process.returncode != 0:
            error_tail = "\n".join(recent_lines[-25:])
            raise RuntimeError(f"Pilot training run exited with non-zero exit code: {process.returncode}\n{error_tail}")

        logger.info("Pilot training run finished successfully.")

        # 4. Locate best or latest pilot checkpoint
        checkpoint_candidates = list(pilot_checkpoint_dir.glob("*.pdparams"))
        if not checkpoint_candidates:
            raise FileNotFoundError(f"No checkpoint .pdparams generated in: {pilot_checkpoint_dir}")

        best_candidates = [c for c in checkpoint_candidates if "best_accuracy" in c.name]
        latest_candidates = [c for c in checkpoint_candidates if "latest" in c.name]
        if best_candidates:
            best_ckpt = best_candidates[0]
        elif latest_candidates:
            best_ckpt = latest_candidates[0]
        else:
            best_ckpt = sorted(checkpoint_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.info(f"Evaluating pilot checkpoint: {best_ckpt}")

        # 5. Benchmark pilot checkpoint against validation set
        engine = BenchmarkEngine(
            config_path=pilot_config_path,
            dict_path=character_dict_path,
        )
        report_path = pilot_dir / "pilot_benchmark_report.json"
        eval_result = engine.run_benchmark(
            model_path=best_ckpt,
            val_label_path=val_label_path,
            max_samples=max_eval_samples,
            output_report_path=report_path,
        )

        return {
            "pilot_checkpoint": str(best_ckpt),
            "report_path": str(report_path),
            "accuracy": eval_result.accuracy,
            "micro_cer": eval_result.micro_cer,
            "micro_wer": eval_result.micro_wer,
        }
