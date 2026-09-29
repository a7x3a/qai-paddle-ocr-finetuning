"""Dynamic YAML configuration injection and hardware-adaptive hyperparameter tuner."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import paddle
import yaml

from .hardware import get_hardware_profile
from .logger import setup_logger

logger = setup_logger("ConfigInjector")


def auto_detect_gpu_batch_size(vram_gb: Optional[float] = None) -> int:
    """Determine optimal training batch size based on available compute hardware.

    Prevents Out-Of-Memory (OOM) failures while maximizing tensor throughput.

    Args:
        vram_gb: Explicit GPU memory in gigabytes. If None, queries hardware profile.

    Returns:
        Recommended batch size integer (multiple of 16 for MultiScaleSampler).
    """
    if vram_gb is not None:
        if vram_gb < 4.0:
            return 32
        elif vram_gb < 6.0:
            return 64
        elif vram_gb < 8.0:
            return 128
        elif vram_gb < 12.0:
            return 256
        elif vram_gb < 16.0:
            return 384
        else:
            return 512

    profile = get_hardware_profile()
    return profile.recommended_train_batch


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
        num_workers: Optional[int] = None,
        epoch_num: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_text_length: int = 40,
        use_gpu: Optional[bool] = None,
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
            use_gpu: Whether to enforce GPU training (None for auto-detect).

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

        # Determine compute device
        has_cuda = bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
        gpu_enabled = has_cuda if use_gpu is None else bool(use_gpu)
        cfg["Global"]["use_gpu"] = gpu_enabled
        if not gpu_enabled:
            cfg["Global"]["amp_level"] = "O0"
        elif "amp_level" not in cfg["Global"]:
            cfg["Global"]["amp_level"] = "O2"

        if pretrained_model_path:
            clean_pretrained = str(Path(pretrained_model_path).resolve())
            if clean_pretrained.endswith(".pdparams"):
                clean_pretrained = clean_pretrained[:-9]
            cfg["Global"]["pretrained_model"] = clean_pretrained
        else:
            cfg["Global"]["pretrained_model"] = None

        if epoch_num is not None:
            cfg["Global"]["epoch_num"] = epoch_num

        cfg["Global"]["print_batch_step"] = 10
        cfg["Global"]["save_epoch_step"] = 1

        # 2. Hardware profile and allocation
        profile = get_hardware_profile()
        effective_bs = batch_size or profile.recommended_train_batch
        effective_workers = num_workers if num_workers is not None else profile.recommended_workers
        logger.info(
            f"Allocated parameters: Batch Size = {effective_bs}, Workers = {effective_workers} "
            f"(Guaranteed Reserved Host RAM: {profile.reserved_free_ram_gb} GB)"
        )

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
                cfg["Train"]["loader"]["num_workers"] = effective_workers
            if "sampler" in cfg["Train"]:
                cfg["Train"]["sampler"]["first_bs"] = effective_bs

        # 4. Eval loader updates
        if "Eval" in cfg:
            cfg["Eval"]["dataset"]["data_dir"] = val_data_dir
            cfg["Eval"]["dataset"]["label_file_list"] = [str(val_p).replace("\\", "/")]
            eval_bs = min(effective_bs, profile.recommended_eval_batch)
            if "loader" in cfg["Eval"]:
                cfg["Eval"]["loader"]["batch_size_per_card"] = eval_bs
                cfg["Eval"]["loader"]["num_workers"] = max(1, effective_workers // 2)

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
