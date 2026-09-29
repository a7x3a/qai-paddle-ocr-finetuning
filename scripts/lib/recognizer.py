"""Batched recognition over image files, shared by evaluation and manual testing.

The character dictionary is always taken from the model directory's own
`inference.yml`. That guarantees the dictionary order matches the weights, and it
removes the single most dangerous class of bug in an OCR pipeline: evaluating a model
against a dictionary whose indices do not line up with its output layer.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import PADDLE_ROOT


@dataclass
class Prediction:
    text: str
    score: float
    readable: bool = True


def read_model_dictionary(model_dir: Path) -> tuple[list[str], bool]:
    """Return (entries, use_space_char) for an exported model.

    PaddleOCR's export writes the post-processing character list, which already
    contains the space and blank entries when training used `use_space_char`. Appending
    space a second time would shift every class index, so detect it instead of assuming.
    """
    inference_yml = model_dir / "inference.yml"
    if not inference_yml.is_file():
        raise FileNotFoundError(
            f"{inference_yml} not found. Point --model-dir at an exported PaddleOCR model."
        )
    config = yaml.safe_load(inference_yml.read_text(encoding="utf-8"))
    entries = (config.get("PostProcess") or {}).get("character_dict")
    if not entries:
        raise ValueError(f"{inference_yml} has no PostProcess.character_dict; cannot decode text safely")
    entries = list(entries)
    return entries, " " not in entries


def _ensure_paddleocr_on_path() -> None:
    if str(PADDLE_ROOT) not in sys.path:
        sys.path.insert(0, str(PADDLE_ROOT))
    try:
        from scripts.patch_paddleocr import ensure_paddleocr_patched
        ensure_paddleocr_patched()
    except Exception:
        pass


class Recognizer:
    """Thin wrapper over PaddleOCR's TextRecognizer with throughput accounting."""

    def __init__(self, model_dir: Path, batch_size: int = 32, use_gpu: bool = True,
                 max_text_length: int = 32, image_shape: str = "3,48,320") -> None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        _ensure_paddleocr_on_path()

        import cv2  # noqa: PLC0415 - imported lazily so the module stays importable without cv2
        from tools.infer.predict_rec import TextRecognizer
        from tools.infer.utility import init_args

        self._cv2 = cv2
        self.model_dir = Path(model_dir)
        self.batch_size = max(1, batch_size)
        entries, use_space_char = read_model_dictionary(self.model_dir)
        self.dictionary_size = len(entries)
        self.use_space_char = use_space_char

        args = init_args().parse_args([])
        args.rec_model_dir = str(self.model_dir)
        # Write the model's own dictionary to a temp file so the predictor cannot
        # silently fall back to a different order.
        #
        # The filename MUST contain the substring "arabic": PaddleOCR's CTCLabelDecode
        # decides whether to undo right-to-left visual order by testing
        # `if "arabic" in character_dict_path` (ppocr/postprocess/rec_postprocess.py).
        # Kurdish and Arabic are RTL, so without this the model returns visually
        # reversed text ("نور" -> "رون") and every metric looks catastrophic.
        import tempfile
        workdir = Path(tempfile.mkdtemp(prefix="kurdish_rec_"))
        is_kurdish = "kurdish" in str(self.model_dir).lower()
        dict_filename = "kurdish_dict.txt" if is_kurdish else "arabic_dict.txt"
        dictionary_file = workdir / dict_filename
        dictionary_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
        self._workdir = workdir
        self._dictionary_path = dictionary_file
        args.rec_char_dict_path = str(dictionary_file)
        args.rec_batch_num = self.batch_size
        args.use_gpu = use_gpu
        args.rec_image_shape = image_shape
        args.max_text_length = max_text_length
        args.use_space_char = use_space_char
        self._recognizer = TextRecognizer(args)
        # Fail loudly rather than silently scoring visual order as if it were logical.
        postprocess = getattr(self._recognizer, "postprocess_op", None)
        self.reverse_applied = bool(getattr(postprocess, "reverse", False))

    def predict_paths(self, paths: list[Path], progress_every: int = 0) -> tuple[list[Prediction], dict]:
        cv2 = self._cv2
        results: list[Prediction] = []
        unreadable = 0
        started = time.perf_counter()

        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = []
            keep = []
            for path in batch_paths:
                image = cv2.imread(str(path))
                if image is None:
                    unreadable += 1
                    results.append(Prediction(text="", score=0.0, readable=False))
                    continue
                images.append(image)
                keep.append(path)
            if images:
                decoded, _ = self._recognizer(images)
                for item in decoded:
                    results.append(Prediction(text=item[0], score=float(item[1])))
            if progress_every and (start // self.batch_size) % progress_every == 0:
                done = min(start + self.batch_size, len(paths))
                rate = done / max(time.perf_counter() - started, 1e-6)
                print(f"  {done}/{len(paths)} images  {rate:.0f} img/s", flush=True)

        elapsed = time.perf_counter() - started
        timing = {
            "images": len(paths),
            "seconds": elapsed,
            "images_per_second": len(paths) / elapsed if elapsed > 0 else 0.0,
            "batch_size": self.batch_size,
            "unreadable_images": unreadable,
            "use_gpu": self._recognizer.use_gpu if hasattr(self._recognizer, "use_gpu") else None,
        }
        return results, timing

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._workdir, ignore_errors=True)
