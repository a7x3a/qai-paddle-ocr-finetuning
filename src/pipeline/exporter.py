"""PaddleOCR model export manager."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union

from src.utils.logger import setup_logger

logger = setup_logger("ModelExporter")


class ModelExporter:
    """Exports trained PaddleOCR recognition models to deployable inference models."""

    def __init__(self, paddleocr_dir: Optional[Union[str, Path]] = None) -> None:
        self.paddleocr_dir = self._find_paddleocr_dir(paddleocr_dir)

    @staticmethod
    def _find_paddleocr_dir(explicit_dir: Optional[Union[str, Path]] = None) -> Path:
        candidates = [
            Path(explicit_dir) if explicit_dir else None,
            Path("./PaddleOCR"),
            Path("../PaddleOCR"),
            Path(__file__).resolve().parent.parent.parent / "PaddleOCR",
        ]
        for c in candidates:
            if c and (c / "tools" / "export_model.py").is_file():
                return c.resolve()
        raise FileNotFoundError("Could not locate PaddleOCR repository containing tools/export_model.py")

    def export(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path],
        output_dir: Union[str, Path],
        dict_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export a trained checkpoint to an inference model directory.

        Args:
            checkpoint_path: Path to checkpoint .pdparams file or checkpoint directory.
            config_path: Path to configuration YAML.
            output_dir: Destination folder for exported inference model.
            dict_path: Optional dictionary path to bundle with the model.

        Returns:
            Resolved Path to exported inference directory.
        """
        ckpt_p = Path(checkpoint_path).resolve()
        cfg_p = Path(config_path).resolve()
        out_p = Path(output_dir).resolve()
        out_p.mkdir(parents=True, exist_ok=True)

        # If ckpt_p is a directory, resolve best_accuracy or latest
        if ckpt_p.is_dir():
            if (ckpt_p / "best_accuracy.pdparams").is_file():
                ckpt_p = ckpt_p / "best_accuracy"
            elif (ckpt_p / "latest.pdparams").is_file():
                ckpt_p = ckpt_p / "latest"
            else:
                candidates = list(ckpt_p.glob("*.pdparams"))
                if candidates:
                    stem = candidates[0].stem
                    ckpt_p = candidates[0].parent / stem
                else:
                    raise FileNotFoundError(f"No .pdparams checkpoint found in: {ckpt_p}")
        else:
            stem = ckpt_p.stem
            if ckpt_p.name.endswith(".pdparams"):
                ckpt_p = ckpt_p.parent / stem

        logger.info(f"Exporting model from checkpoint: {ckpt_p}")
        logger.info(f"Using configuration: {cfg_p}")
        logger.info(f"Output inference directory: {out_p}")

        export_py = self.paddleocr_dir / "tools" / "export_model.py"
        cmd = [
            sys.executable,
            str(export_py),
            "-c",
            str(cfg_p),
            "-o",
            f"Global.checkpoints={str(ckpt_p).replace('\\', '/')}",
            f"Global.save_inference_dir={str(out_p).replace('\\', '/')}",
            "Global.infer_mode=false",
            "Global.use_space_char=true",
        ]

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
                if "inference model is saved" in line_str or "export" in line_str.lower():
                    logger.info(f"[EXPORT] {line_str}")

        process.wait()
        if process.returncode != 0:
            tail = "\n".join(recent_lines[-20:])
            raise RuntimeError(f"Export failed with exit code {process.returncode}:\n{tail}")

        # Bundle character dictionary so the exported model is self-contained
        if dict_path and Path(dict_path).is_file():
            shutil.copy2(str(dict_path), str(out_p / Path(dict_path).name))
        elif Path("configs/arabic_kurdish_dict.txt").is_file():
            shutil.copy2("configs/arabic_kurdish_dict.txt", str(out_p / "arabic_kurdish_dict.txt"))

        exported_files = sorted(list(out_p.iterdir()), key=lambda x: x.name)
        logger.info(f"Export successfully completed! {len(exported_files)} files written to: {out_p}")
        for f in exported_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name:<25} ({size_mb:6.2f} MB)")

        return out_p
