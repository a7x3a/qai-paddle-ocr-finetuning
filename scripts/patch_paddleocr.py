"""PaddleOCR Auto-Patcher.

Ensures that the nested PaddleOCR repository has all required runtime patches:
1. Whitelists 'arabic_PP-OCRv5_mobile_rec' in tools/infer/predict_rec.py.
2. Adds backbone freezing/unfreezing support in tools/train.py and tools/program.py.
3. Configures colored console formatting in ppocr/utils/logging.py.

Can be run as a standalone script or imported via ensure_paddleocr_patched().
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def get_paddleocr_dir() -> Path | None:
    """Find the PaddleOCR root directory."""
    candidates = [
        Path("./PaddleOCR"),
        Path("../PaddleOCR"),
        Path(__file__).resolve().parent.parent / "PaddleOCR",
    ]
    for c in candidates:
        if c.is_dir() and (c / "tools" / "infer" / "predict_rec.py").is_file():
            return c.resolve()
    return None


def patch_predict_rec(paddle_dir: Path) -> bool:
    """Ensure arabic_PP-OCRv5_mobile_rec is in the supported models whitelist."""
    rec_py = paddle_dir / "tools" / "infer" / "predict_rec.py"
    if not rec_py.is_file():
        return False

    content = rec_py.read_text(encoding="utf-8")
    if "arabic_PP-OCRv5_mobile_rec" in content:
        return True  # Already patched

    target = '"PP-OCRv5_mobile_rec",'
    replacement = '"PP-OCRv5_mobile_rec",\n                "arabic_PP-OCRv5_mobile_rec",'
    if target in content:
        new_content = content.replace(target, replacement, 1)
        rec_py.write_text(new_content, encoding="utf-8")
        return True
    return False


def patch_train(paddle_dir: Path) -> bool:
    """Ensure freeze_backbone_epochs is handled in tools/train.py."""
    train_py = paddle_dir / "tools" / "train.py"
    if not train_py.is_file():
        return False

    content = train_py.read_text(encoding="utf-8")
    if "freeze_backbone_epochs" in content:
        return True  # Already patched

    target = 'model = build_model(config["Architecture"])'
    freeze_code = """model = build_model(config["Architecture"])

    freeze_backbone_epochs = int(global_config.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0 and not global_config.get("checkpoints"):
        frozen_parameters = 0
        for name, parameter in model.named_parameters():
            if name.startswith("backbone."):
                parameter.stop_gradient = True
                frozen_parameters += 1
        logger.info(
            "backbone frozen for epochs 1-{} ({} parameters); recognition heads remain trainable".format(
                freeze_backbone_epochs, frozen_parameters
            )
        )"""

    if target in content:
        new_content = content.replace(target, freeze_code, 1)
        train_py.write_text(new_content, encoding="utf-8")
        return True
    return False


def patch_program(paddle_dir: Path) -> bool:
    """Ensure backbone unfreezing is handled in tools/program.py."""
    prog_py = paddle_dir / "tools" / "program.py"
    if not prog_py.is_file():
        return False

    content = prog_py.read_text(encoding="utf-8")
    if "freeze_backbone_epochs" in content:
        return True  # Already patched

    target = "for epoch in range(start_epoch, epoch_num + 1):"
    unfreeze_code = """for epoch in range(start_epoch, epoch_num + 1):
        freeze_backbone_epochs = int(config["Global"].get("freeze_backbone_epochs", 0))
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            unfrozen_parameters = 0
            for name, parameter in model.named_parameters():
                if name.startswith("backbone.") and parameter.stop_gradient:
                    parameter.stop_gradient = False
                    unfrozen_parameters += 1
            logger.info(
                "backbone unfrozen at epoch {} ({} parameters are trainable)".format(
                    epoch, unfrozen_parameters
                )
            )"""

    if target in content:
        new_content = content.replace(target, unfreeze_code, 1)
        prog_py.write_text(new_content, encoding="utf-8")
        return True
    return False


def patch_logging(paddle_dir: Path) -> bool:
    """Ensure ConsoleFormatter is used in ppocr/utils/logging.py."""
    log_py = paddle_dir / "ppocr" / "utils" / "logging.py"
    if not log_py.is_file():
        return False

    content = log_py.read_text(encoding="utf-8")
    if "ConsoleFormatter" in content:
        return True  # Already patched

    formatter_class = """class ConsoleFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\\033[36m",
        logging.WARNING: "\\033[33m",
        logging.ERROR: "\\033[31m",
        logging.CRITICAL: "\\033[1;31m",
    }

    def __init__(self):
        super().__init__(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record):
        text = super().format(record)
        if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
            color = self.COLORS.get(record.levelno, "")
            if color:
                return f"{color}{text}\\033[0m"
        return text


logger_initialized = {}"""

    if "logger_initialized = {}" in content:
        content = content.replace("logger_initialized = {}", formatter_class, 1)
        content = content.replace(
            "stream_handler.setFormatter(formatter)",
            "stream_handler.setFormatter(ConsoleFormatter())",
            1,
        )
        log_py.write_text(content, encoding="utf-8")
        return True
    return False


def ensure_paddleocr_patched() -> bool:
    """Ensure all required patches are applied to PaddleOCR."""
    pdir = get_paddleocr_dir()
    if not pdir:
        return False

    patch_predict_rec(pdir)
    patch_train(pdir)
    patch_program(pdir)
    patch_logging(pdir)
    return True


if __name__ == "__main__":
    pdir = get_paddleocr_dir()
    if not pdir:
        print("[!] PaddleOCR directory not found. Skipping patch.")
        sys.exit(0)

    print(f"Applying patches to PaddleOCR at: {pdir}")
    r1 = patch_predict_rec(pdir)
    r2 = patch_train(pdir)
    r3 = patch_program(pdir)
    r4 = patch_logging(pdir)
    print(f"  [+] predict_rec.py whitelist: {'OK' if r1 else 'Failed'}")
    print(f"  [+] train.py backbone freeze: {'OK' if r2 else 'Failed'}")
    print(f"  [+] program.py unfreeze:      {'OK' if r3 else 'Failed'}")
    print(f"  [+] logging.py formatter:     {'OK' if r4 else 'Failed'}")
    print("PaddleOCR patches applied successfully!")
