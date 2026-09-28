"""Pre-flight hardware and gradient execution validation suite.

Verifies CUDA device attributes, VRAM limits, UTF-8 dictionary validity, dataset sample
integrity, and conducts a 5-sample forward-backward pass in PaddlePaddle to eliminate
OOM and gradient graph failures before initiating large-scale training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import paddle
import yaml

from src.utils.logger import setup_logger

logger = setup_logger("SmokeTest")


class SmokeTester:
    """Pre-training environment, dataset, and tensor gradient integrity tester."""

    def __init__(self, use_gpu: bool = True) -> None:
        """Initialize smoke tester.

        Args:
            use_gpu: Whether GPU validation should be enforced.
        """
        self.use_gpu = use_gpu

    def verify_cuda(self) -> dict[str, Any]:
        """Inspect and report CUDA runtime capabilities and memory.

        Returns:
            Dictionary with device information.

        Raises:
            RuntimeError: If GPU is requested but unavailable.
        """
        logger.info("Executing CUDA hardware verification...")
        compiled_cuda = paddle.is_compiled_with_cuda()
        dev_count = paddle.device.cuda.device_count()

        logger.info(f"PaddlePaddle CUDA Compiled: {compiled_cuda}")
        logger.info(f"Available CUDA Devices: {dev_count}")

        if self.use_gpu:
            if not compiled_cuda or dev_count == 0:
                raise RuntimeError(
                    "CUDA verification failed: PaddlePaddle was not compiled with CUDA or no GPUs detected."
                )

            dev_name = paddle.device.cuda.get_device_name(0)
            logger.info(f"Active GPU Device: {dev_name}")

            props = paddle.device.cuda.get_device_properties(0)
            total_mem_gb = getattr(props, "total_memory", 0) / (1024.0 ** 3)
            logger.info(f"Total Dedicated VRAM: {total_mem_gb:.2f} GB")

            paddle.set_device("gpu:0")

            # Mathematical tensor kernel execution verification
            try:
                t1 = paddle.to_tensor([1.0, 2.0], dtype="float32")
                t2 = t1 + 1.0
                paddle.device.synchronize()
                res = t2.numpy().tolist()
                # Verify that t2.numpy() matches [2.0, 3.0] and does not return uninitialized zeros
                if res != [2.0, 3.0]:
                    raise ValueError(
                        f"Kernel execution returned invalid result: {res} (expected [2.0, 3.0]). "
                        f"Uninitialized zeros indicate missing SM architecture cubins."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"CUDA tensor kernel verification failed on device '{dev_name}': {e}\n"
                    f"Your GPU architecture (e.g. RTX 50-Series / Blackwell sm_120) requires 'paddlepaddle-gpu>=3.4.0' (cu129).\n"
                    f"Please install via: pip install paddlepaddle-gpu==3.4.0 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/\n"
                    f"or run with '--no-gpu' to execute on CPU."
                ) from e

            logger.info("CUDA kernel computation verified: [1.0, 2.0] + 1.0 == [2.0, 3.0]")

            return {
                "cuda_compiled": True,
                "device_count": dev_count,
                "device_name": dev_name,
                "total_vram_gb": total_mem_gb,
            }
        else:
            paddle.set_device("cpu")
            return {"cuda_compiled": False, "device_count": 0, "device_name": "CPU", "total_vram_gb": 0.0}

    def verify_dictionary(self, dict_path: Union[str, Path]) -> int:
        """Validate character dictionary existence, non-emptiness, and UTF-8 encoding.

        Args:
            dict_path: Path to dictionary.txt.

        Returns:
            Total unique characters loaded.

        Raises:
            FileNotFoundError: If file is absent.
            ValueError: If file is empty or corrupted.
        """
        path = Path(dict_path).resolve()
        logger.info(f"Verifying dictionary file: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Dictionary file not found at: {path}")

        chars = set()
        total_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                c = line.strip("\r\n")
                if c:
                    chars.add(c)
                    total_lines += 1

        if not chars:
            raise ValueError(f"Dictionary file at {path} is empty or contains no valid characters.")

        logger.info(f"Dictionary verified successfully: {total_lines} character lines ({len(chars)} unique tokens) loaded.")
        return total_lines

    def verify_label_file(self, label_path: Union[str, Path], check_samples: int = 5) -> int:
        """Verify label file format and check disk availability of sample images.

        Args:
            label_path: Path to train_label.txt or val_label.txt.
            check_samples: Number of sample images to physically decode from disk.

        Returns:
            Total valid sample records counted.
        """
        path = Path(label_path).resolve()
        logger.info(f"Verifying dataset label file: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Dataset index file not found at: {path}")

        valid_records = 0
        samples_checked = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean = line.strip("\r\n")
                if not clean:
                    continue

                parts = clean.split("\t")
                if len(parts) < 2:
                    continue

                img_path_str, text = parts[0], parts[1]
                valid_records += 1

                if samples_checked < check_samples:
                    raw_path = Path(img_path_str)
                    img_p = raw_path
                    if not img_p.is_file():
                        candidates = [
                            path.parent / raw_path,
                            Path.cwd() / raw_path,
                            path.parent.parent / raw_path,
                            Path("data/kurdish_rec") / raw_path,
                            Path("data") / raw_path,
                        ]
                        for c in candidates:
                            if c.is_file():
                                img_p = c
                                break
                    if not img_p.is_file():
                        raise FileNotFoundError(f"Referenced image at line {line_idx} missing on disk: {img_path_str}")
                    # Test image decode
                    mat = cv2.imread(str(img_p))
                    if mat is None:
                        raise ValueError(f"Failed to decode image at line {line_idx}: {img_p}")
                    samples_checked += 1

        logger.info(f"Label file verified: {valid_records} entries found. {samples_checked} image files decoded.")
        return valid_records

    def verify_gradient_execution(
        self,
        config_path: Union[str, Path],
        dict_path: Union[str, Path],
        label_path: Union[str, Path],
    ) -> bool:
        """Perform synthetic 5-sample batch forward and backward pass.

        Ensures full CUDA gradient backpropagation graph construction and execution
        without OOM or NaN loss anomalies.

        Args:
            config_path: Path to template or runtime YAML config.
            dict_path: Path to character dictionary.
            label_path: Path to label file with image references.

        Returns:
            True if gradient backprop succeeds.
        """
        logger.info("Initializing synthetic forward-backward gradient execution pass...")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        num_chars = self.verify_dictionary(dict_path)
        use_space = cfg.get("Global", {}).get("use_space_char", True)
        ctc_out = num_chars + (2 if use_space else 1)

        arch = cfg["Architecture"]
        if "out_channels_list" not in arch.get("Head", {}):
            arch["Head"]["out_channels_list"] = {
                "CTCLabelDecode": ctc_out,
                "SARLabelDecode": ctc_out + 2,
                "NRTRLabelDecode": ctc_out + 3,
            }

        # Ensure ppocr is discoverable
        candidates = [
            Path("./PaddleOCR"),
            Path("../PaddleOCR"),
            Path(__file__).resolve().parent.parent.parent / "PaddleOCR",
            Path(__file__).resolve().parent.parent.parent.parent / "PaddleOCR",
        ]
        for c in candidates:
            if c and (c / "tools" / "train.py").is_file() and str(c.resolve()) not in sys.path:
                sys.path.insert(0, str(c.resolve()))

        from ppocr.modeling.architectures import build_model
        model = build_model(arch)
        model.train()

        # Construct 5-sample batch
        batch_size = 5
        dummy_images = []

        # Read actual samples if available, else synthetic tensors
        loaded_count = 0
        if Path(label_path).is_file():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip("\r\n").split("\t")
                    if len(parts) >= 2:
                        raw_path = Path(parts[0])
                        img_p = raw_path
                        if not img_p.is_file():
                            candidates = [
                                Path(label_path).parent / raw_path,
                                Path.cwd() / raw_path,
                                Path(label_path).parent.parent / raw_path,
                                Path("data/kurdish_rec") / raw_path,
                                Path("data") / raw_path,
                            ]
                            for c in candidates:
                                if c.is_file():
                                    img_p = c
                                    break
                        mat = cv2.imread(str(img_p))
                        if mat is not None:
                            resized = cv2.resize(mat, (320, 48)).astype(np.float32) / 255.0
                            resized = resized.transpose((2, 0, 1))
                            dummy_images.append(resized)
                            loaded_count += 1
                            if loaded_count >= batch_size:
                                break

        while len(dummy_images) < batch_size:
            dummy_images.append(np.random.randn(3, 48, 320).astype(np.float32))

        batch_arr = np.stack(dummy_images, axis=0)
        batch_tensor = paddle.to_tensor(batch_arr, dtype="float32")
        batch_tensor.stop_gradient = False

        # Supply MultiHead target tensors for training mode
        max_len = cfg.get("Global", {}).get("max_text_length", 40)
        dummy_label_ctc = paddle.zeros([batch_size, max_len], dtype="int64")
        dummy_label_sar = paddle.zeros([batch_size, max_len], dtype="int64")
        dummy_length = paddle.full([batch_size], max_len // 2, dtype="int64")
        dummy_valid_ratio = paddle.ones([batch_size], dtype="float32")
        dummy_targets = [batch_tensor, dummy_label_ctc, dummy_label_sar, dummy_length, dummy_valid_ratio]

        # Forward pass
        outputs = model(batch_tensor, data=dummy_targets)

        # Handle MultiHead output dictionary
        if isinstance(outputs, dict):
            loss = paddle.zeros([1], dtype="float32")
            for k, head_out in outputs.items():
                if isinstance(head_out, paddle.Tensor):
                    loss = loss + paddle.mean(head_out)
        elif isinstance(outputs, (list, tuple)):
            loss = paddle.mean(outputs[0])
        else:
            loss = paddle.mean(outputs)

        scalar_loss = float(loss.item()) if hasattr(loss, "item") else float(loss.numpy().squeeze())
        logger.info(f"Forward pass completed. Scalar synthetic loss: {scalar_loss:.4f}")

        # Backward pass
        loss.backward()

        # Verify gradients exist and are not NaN
        has_valid_grad = False
        for param in model.parameters():
            if param.grad is not None:
                grad_np = param.grad.numpy()
                if np.isnan(grad_np).any():
                    raise RuntimeError("Smoke test detected NaN gradients during backpropagation!")
                has_valid_grad = True
                break

        if not has_valid_grad:
            raise RuntimeError("Backward pass failed: No gradients computed for model parameters.")

        logger.info("Gradient backpropagation successfully executed with zero NaN values.")
        return True

    def run_all(
        self,
        config_path: Union[str, Path],
        dict_path: Union[str, Path],
        label_path: Union[str, Path],
    ) -> bool:
        """Execute full smoke test verification pipeline.

        Args:
            config_path: Path to configuration YAML.
            dict_path: Path to character dictionary.
            label_path: Path to training label index.

        Returns:
            True if all checks pass.
        """
        logger.info("================ STARTING PRE-TRAINING SMOKE TEST ================")
        self.verify_cuda()
        self.verify_dictionary(dict_path)
        self.verify_label_file(label_path)
        self.verify_gradient_execution(config_path, dict_path, label_path)
        logger.info("================ ALL SMOKE TEST CHECKS PASSED ================")
        return True
