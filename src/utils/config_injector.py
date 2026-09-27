"""Dynamic YAML configuration injection and hardware-adaptive hyperparameter tuner."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import paddle
import yaml

from .logger import setup_logger

logger = setup_logger("ConfigInjector")


def auto_detect_gpu_batch_size(vram_gb: Optional[float] = None) -> int:
    """Determine optimal training batch size based on available CUDA VRAM.

    Prevents Out-Of-Memory (OOM) failures while maximizing tensor throughput.

    Args:
        vram_gb: Explicit GPU memory in gigabytes. If None, queries active CUDA device.

    Returns:
        Recommended batch size integer (e.g. 32, 64, 128, 256, 384).
    """
    if vram_gb is None:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            try:
                # Query device 0 memory capability
                dev_id = 0
                props = paddle.device.cuda.get_device_properties(dev_id)
                # Total memory in bytes to GB
                total_bytes = getattr(props, "total_memory", 0)
                vram_gb = total_bytes / (1024.0 ** 3)
            except Exception:
                vram_gb = 6.0
        else:
            return 16  # CPU fallback

    if vram_gb < 4.0:
        return 32
    elif vram_gb < 6.0:
        return 64
    elif vram_gb < 8.0:
        return 128
    elif vram_gb < 12.0:
        return 256
    else:
        return 384


class PaddleConfigInjector:
    """Manages programmatic modification and serialization of PaddleOCR configuration trees."""

    def __init__(self, template_config_path: Union[str, Path]) -> None:
        """Load baseline template YAML configuration.

        Args:
            template_config_path: Path to template_rec_v4.yml.
        """
        self.template_path = Path(template_config_path).resolve()
        if not self.template_path.is_file():
            raise FileNotFoundError(f"Template config does not exist: {self.template_path}")

        with open(self.template_path, "r", encoding="utf-8") as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

    def inject_runtime_paths(
        self,
        character_dict_path: Union[str, Path],
        train_label_path: Union[str, Path],
        val_label_path: Union[str, Path],
        save_model_dir: Union[str, Path],
        pretrained_model_path: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 8,
        epoch_num: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_text_length: int = 40,
    ) -> dict[str, Any]:
        """Inject runtime paths and hardware-adaptive parameters into the configuration.

        Args:
            character_dict_path: Path to vocabulary dictionary.txt.
            train_label_path: Path to train_label.txt.
            val_label_path: Path to val_label.txt.
            save_model_dir: Directory where model checkpoints will be written.
            pretrained_model_path: Optional base weights path without suffix.
            batch_size: Explicit batch size per GPU. If None, auto-detected from VRAM.
            num_workers: DataLoader worker thread/process count.
            epoch_num: Optional override for total training epochs.
            learning_rate: Optional override for initial learning rate.
            max_text_length: Maximum sequence length for recognition heads.

        Returns:
            Updated configuration dictionary.
        """
        cfg = copy.deepcopy(self.config)

        dict_p = Path(character_dict_path).resolve()
        train_p = Path(train_label_path).resolve()
        val_p = Path(val_label_path).resolve()
        save_dir_p = Path(save_model_dir).resolve()
        save_dir_p.mkdir(parents=True, exist_ok=True)

        # Count character lines in dictionary
        num_dict_chars = 0
        if dict_p.is_file():
            with open(dict_p, "r", encoding="utf-8") as f:
                num_dict_chars = sum(1 for line in f if line.strip("\r\n"))

        # In PaddleOCR CTC: classes = dict_chars + 1 (blank) + 1 (space if use_space_char)
        use_space = cfg.get("Global", {}).get("use_space_char", True)
        ctc_classes = num_dict_chars + (2 if use_space else 1)
        sar_classes = num_dict_chars + (2 if use_space else 1)

        # 1. Global updates
        cfg["Global"]["character_dict_path"] = str(dict_p)
        cfg["Global"]["save_model_dir"] = str(save_dir_p)
        cfg["Global"]["max_text_length"] = max_text_length
        cfg["Global"]["save_res_path"] = str(save_dir_p / "predicts.txt")
        cfg["Global"]["save_inference_dir"] = str(save_dir_p / "inference")

        if pretrained_model_path:
            clean_pretrained = str(Path(pretrained_model_path).resolve())
            if clean_pretrained.endswith(".pdparams"):
                clean_pretrained = clean_pretrained[:-9]
            cfg["Global"]["pretrained_model"] = clean_pretrained
        else:
            cfg["Global"]["pretrained_model"] = None

        if epoch_num is not None:
            cfg["Global"]["epoch_num"] = epoch_num

        # 2. Batch size resolution
        effective_bs = batch_size or auto_detect_gpu_batch_size()
        logger.info(f"Target batch size allocated: {effective_bs}")

        # Determine dataset root containing 'images'
        train_data_dir = "./"
        if (train_p.parent / "images").is_dir():
            train_data_dir = str(train_p.parent).replace("\\", "/") + "/"
        elif (Path("data/kurdish_rec") / "images").is_dir():
            train_data_dir = str(Path("data/kurdish_rec").resolve()).replace("\\", "/") + "/"

        val_data_dir = "./"
        if (val_p.parent / "images").is_dir():
            val_data_dir = str(val_p.parent).replace("\\", "/") + "/"
        elif (Path("data/kurdish_rec") / "images").is_dir():
            val_data_dir = str(Path("data/kurdish_rec").resolve()).replace("\\", "/") + "/"

        # 3. Train loader updates
        if "Train" in cfg:
            cfg["Train"]["dataset"]["data_dir"] = train_data_dir
            cfg["Train"]["dataset"]["label_file_list"] = [str(train_p).replace("\\", "/")]
            if "loader" in cfg["Train"]:
                cfg["Train"]["loader"]["batch_size_per_card"] = effective_bs
                cfg["Train"]["loader"]["num_workers"] = num_workers
            if "sampler" in cfg["Train"]:
                cfg["Train"]["sampler"]["first_bs"] = effective_bs

        # 4. Eval loader updates
        if "Eval" in cfg:
            cfg["Eval"]["dataset"]["data_dir"] = val_data_dir
            cfg["Eval"]["dataset"]["label_file_list"] = [str(val_p).replace("\\", "/")]
            eval_bs = min(effective_bs, 64)
            if "loader" in cfg["Eval"]:
                cfg["Eval"]["loader"]["batch_size_per_card"] = eval_bs
                cfg["Eval"]["loader"]["num_workers"] = max(2, num_workers // 2)

        # 5. Architecture Head classes
        if "Architecture" in cfg and "Head" in cfg["Architecture"]:
            cfg["Architecture"]["Head"]["out_channels_list"] = {
                "CTCLabelDecode": ctc_classes,
                "SARLabelDecode": sar_classes,
                "NRTRLabelDecode": ctc_classes + 3,
            }

        # 6. Optimizer LR update
        if learning_rate is not None and "Optimizer" in cfg and "lr" in cfg["Optimizer"]:
            cfg["Optimizer"]["lr"]["learning_rate"] = learning_rate

        self.config = cfg
        return cfg

    def write_runtime_config(self, output_config_path: Union[str, Path]) -> Path:
        """Serialize runtime configuration tree to YAML.

        Args:
            output_config_path: Destination path for runtime_config.yml.

        Returns:
            Resolved Path of the saved file.
        """
        output_path = Path(output_config_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated runtime config written to: {output_path}")
        return output_path
