"""Benchmarking engine for PaddleOCR recognition models and checkpoints.

Provides automated evaluation across raw pretrained models, exported inference models,
and training checkpoints against validation label files. Measures character/word error rates,
exact-match accuracy, latency, and throughput, producing structured JSON reports.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import paddle
import yaml

from src.metrics.evaluator import EvaluationResult, MetricsEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("BenchmarkEngine")


class CTCDecoder:
    """Greedy CTC decoder with support for custom UTF-8 character dictionaries."""

    def __init__(self, character_dict_path: Union[str, Path], use_space_char: bool = True) -> None:
        """Initialize dictionary and reverse mapping.

        Args:
            character_dict_path: Path to character dictionary file (one char per line).
            use_space_char: Whether space character is included in vocabulary.
        """
        self.character_dict_path = Path(character_dict_path)
        self.character_list: list[str] = []
        if self.character_dict_path.is_file():
            with open(self.character_dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    c = line.strip("\r\n")
                    if c:
                        self.character_list.append(c)

        if use_space_char:
            self.character_list.append(" ")

        # PaddleOCR CTC dictionary structure: 0 is CTC blank, followed by dict chars
        self.character_list = ["blank"] + self.character_list
        self.reverse = "arabic" in str(self.character_dict_path).lower()

    def pred_reverse(self, chars: list[str]) -> str:
        """Reverse RTL character sequence adhering to official PaddleOCR CTCLabelDecode rules."""
        import re
        pred_re = []
        c_current = ""
        for c in chars:
            if not bool(re.search(r"[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)
        return "".join(pred_re[::-1])

    def decode(self, preds: np.ndarray) -> list[tuple[str, float]]:
        """Greedy CTC decode from output probabilities or logits.

        Args:
            preds: Array of shape (B, T, C) or (B, T).

        Returns:
            List of (text, confidence) tuples for each batch sample.
        """
        if len(preds.shape) == 3:
            pred_indices = np.argmax(preds, axis=-1)
            pred_probs = np.max(preds, axis=-1)
        else:
            pred_indices = preds
            pred_probs = np.ones_like(preds, dtype=np.float32)

        results: list[tuple[str, float]] = []
        for b_idx in range(len(pred_indices)):
            indices = pred_indices[b_idx]
            probs = pred_probs[b_idx]
            chars: list[str] = []
            char_probs: list[float] = []
            prev_idx = -1

            for step, idx in enumerate(indices):
                if idx != 0 and idx != prev_idx:
                    if idx < len(self.character_list):
                        chars.append(self.character_list[idx])
                        char_probs.append(float(probs[step]))
                prev_idx = idx

            text = self.pred_reverse(chars) if self.reverse else "".join(chars)
            conf = float(np.mean(char_probs)) if char_probs else 0.0
            results.append((text, conf))

        return results


class BenchmarkEngine:
    """Evaluates PaddleOCR checkpoints or exported models on target test datasets."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        dict_path: Optional[Union[str, Path]] = None,
        use_gpu: bool = True,
    ) -> None:
        """Initialize benchmark engine.

        Args:
            config_path: Path to YAML configuration for model architecture.
            dict_path: Path to character dictionary.
            use_gpu: Whether to utilize GPU acceleration.
        """
        self.config_path = Path(config_path) if config_path else None
        self.dict_path = Path(dict_path) if dict_path else None
        self.use_gpu = use_gpu and paddle.is_compiled_with_cuda()
        self.evaluator = MetricsEvaluator()
        self.decoder: Optional[CTCDecoder] = None

        if self.dict_path and self.dict_path.is_file():
            self.decoder = CTCDecoder(self.dict_path)

        if self.use_gpu:
            paddle.set_device("gpu:0")
        else:
            paddle.set_device("cpu")

    def _preprocess_image(
        self,
        image_path: Union[str, Path],
        target_shape: tuple[int, int, int] = (3, 48, 320),
    ) -> Optional[np.ndarray]:
        """Read and resize image preserving aspect ratio with right-padding.

        Args:
            image_path: Path to source image.
            target_shape: (C, H, W).

        Returns:
            Normalized float32 array shaped (C, H, W) or None on failure.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        c, h, w = target_shape
        img_h, img_w, _ = img.shape
        ratio = img_w / float(img_h)
        resized_w = int(math.ceil(h * ratio)) if "math" in globals() else int(round(h * ratio))
        resized_w = max(min(resized_w, w), 8)

        resized = cv2.resize(img, (resized_w, h), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32)
        resized = resized / 255.0
        resized -= 0.5
        resized /= 0.5

        # Padding to max width
        padding_im = np.zeros((h, w, c), dtype=np.float32)
        padding_im[:, :resized_w, :] = resized
        # Transpose to (C, H, W)
        padding_im = padding_im.transpose((2, 0, 1))
        return padding_im

    def _load_model(
        self,
        model_path: Union[str, Path],
        config: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Load model dynamically from config or as inference model.

        Args:
            model_path: Path to checkpoint .pdparams or directory containing inference files.
            config: Parsed configuration dictionary.

        Returns:
            Instantiated model in eval mode or Paddle predictor.
        """
        model_path = Path(model_path)

        # Case 1: Directory with exported inference model (paddle.inference)
        if model_path.is_dir():
            model_file = (
                model_path / "inference.json"
                if (model_path / "inference.json").is_file()
                else model_path / "inference.pdmodel"
            )
            params_file = model_path / "inference.pdiparams"
            if model_file.is_file() and params_file.is_file():
                import paddle.inference as paddle_infer

                infer_config = paddle_infer.Config(
                    str(model_file),
                    str(params_file),
                )
                if self.use_gpu:
                    infer_config.enable_use_gpu(500, 0)
                else:
                    infer_config.disable_gpu()
                infer_config.enable_memory_optim()
                predictor = paddle_infer.create_predictor(infer_config)
                logger.info(f"Successfully loaded inference predictor from: {model_path}")
                return ("inference", predictor)

        # Case 2: PyTorch/Paddle Dynamic Checkpoint (.pdparams)
        from ppocr.modeling.architectures import build_model

        if config is None:
            if self.config_path and self.config_path.is_file():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError("Config required to instantiate dynamic architecture for .pdparams")

        # Set character out_channels if needed
        arch = config["Architecture"]
        if "out_channels_list" not in arch.get("Head", {}):
            num_classes = len(self.decoder.character_list) if self.decoder else 749
            nrtr_classes = num_classes + 3
            arch["Head"]["out_channels_list"] = {
                "CTCLabelDecode": num_classes,
                "SARLabelDecode": num_classes + 2,
                "NRTRLabelDecode": nrtr_classes,
            }

        model = build_model(arch)
        model.eval()

        # Load weights
        weight_file = model_path
        if weight_file.is_dir():
            candidates = list(weight_file.glob("*.pdparams"))
            if candidates:
                weight_file = candidates[0]

        param_path_str = str(weight_file)
        if param_path_str.endswith(".pdparams"):
            param_path_str = param_path_str[:-9]

        state_dict = paddle.load(param_path_str + ".pdparams")
        model.set_state_dict(state_dict)
        logger.info(f"Successfully loaded checkpoint weights from: {weight_file}")
        return ("dynamic", model)

    def run_benchmark(
        self,
        model_path: Union[str, Path],
        val_label_path: Union[str, Path],
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        output_report_path: Optional[Union[str, Path]] = None,
    ) -> EvaluationResult:
        """Execute benchmark inference against a validation dataset.

        Args:
            model_path: Path to model checkpoint or directory.
            val_label_path: Path to val_label.txt containing image_path<TAB>text.
            batch_size: Evaluation batch size.
            max_samples: Optional maximum number of samples to evaluate.
            output_report_path: Optional destination to write JSON benchmark report.

        Returns:
            EvaluationResult containing accuracy, CER, WER, and latency metrics.
        """
        model_path = Path(model_path)
        val_label_path = Path(val_label_path)

        if not val_label_path.is_file():
            raise FileNotFoundError(f"Validation label file not found: {val_label_path}")

        # Load pairs
        records: list[tuple[Path, str]] = []
        with open(val_label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip("\r\n").split("\t")
                if len(parts) >= 2:
                    raw_path = Path(parts[0])
                    img_p = raw_path
                    if not img_p.is_file():
                        candidates = [
                            val_label_path.parent / raw_path,
                            Path.cwd() / raw_path,
                            val_label_path.parent.parent / raw_path,
                            Path("data/kurdish_rec") / raw_path,
                            Path("data") / raw_path,
                        ]
                        for c in candidates:
                            if c.is_file():
                                img_p = c
                                break
                    records.append((img_p, parts[1]))
                    if max_samples is not None and len(records) >= max_samples:
                        break

        logger.info(f"Loaded {len(records)} test samples for benchmarking")
        if not records:
            return self.evaluator.compute()

        model_type, model = self._load_model(model_path)
        self.evaluator.reset()

        start_time = time.perf_counter()

        # Batch processing loop
        for i in range(0, len(records), batch_size):
            batch_slice = records[i : i + batch_size]
            batch_images: list[np.ndarray] = []
            batch_refs: list[str] = []

            for img_p, gt_text in batch_slice:
                prep = self._preprocess_image(img_p)
                if prep is not None:
                    batch_images.append(prep)
                    batch_refs.append(gt_text)

            if not batch_images:
                continue

            batch_tensor = np.stack(batch_images, axis=0)
            sample_start = time.perf_counter()

            if model_type == "inference":
                input_names = model.get_input_names()
                input_handle = model.get_input_handle(input_names[0])
                input_handle.reshape(batch_tensor.shape)
                input_handle.copy_from_cpu(batch_tensor)
                model.run()
                output_names = model.get_output_names()
                output_handle = model.get_output_handle(output_names[0])
                preds = output_handle.copy_to_cpu()
            else:
                with paddle.no_grad():
                    paddle_tensor = paddle.to_tensor(batch_tensor)
                    out = model(paddle_tensor)
                    if isinstance(out, dict):
                        preds = out.get("ctc", list(out.values())[0]).numpy()
                    elif isinstance(out, (list, tuple)):
                        preds = out[0].numpy()
                    else:
                        preds = out.numpy()

            sample_time_ms = ((time.perf_counter() - sample_start) * 1000.0) / len(batch_refs)

            # CTC Decode
            if self.decoder:
                decoded_preds = self.decoder.decode(preds)
                pred_texts = [p[0] for p in decoded_preds]
            else:
                pred_texts = ["" for _ in range(len(batch_refs))]

            latencies = [sample_time_ms] * len(batch_refs)
            self.evaluator.update_batch(batch_refs, pred_texts, latencies)

        total_elapsed = time.perf_counter() - start_time
        result = self.evaluator.compute(total_elapsed_sec=total_elapsed)

        logger.info(
            f"Benchmark Result: CER(micro)={result.micro_cer:.4f} | WER(micro)={result.micro_wer:.4f} | "
            f"Acc={result.accuracy:.4f} | Latency={result.avg_latency_ms:.2f}ms | FPS={result.throughput_fps:.1f}"
        )

        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            report_data = {
                "checkpoint_path": str(model_path.resolve()),
                "total_samples": result.total_samples,
                "exact_matches": result.exact_matches,
                "accuracy": round(result.accuracy, 4),
                "macro_cer": round(result.macro_cer, 4),
                "micro_cer": round(result.micro_cer, 4),
                "macro_wer": round(result.macro_wer, 4),
                "micro_wer": round(result.micro_wer, 4),
                "avg_latency_ms": round(result.avg_latency_ms, 2),
                "throughput_fps": round(result.throughput_fps, 2),
                "total_time_sec": round(result.total_time_sec, 2),
            }
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Wrote benchmark report to: {out_p}")

        return result
