"""Dataset records, dictionary construction, and pretrained-head verification.

The dictionary is the single most important decision in this project. The base
`arabic_PP-OCRv5_mobile_rec` checkpoint has a 749-class CTC head, which corresponds
exactly to its own 747-entry character dictionary plus the space character that
PaddleOCR appends when `use_space_char: true`. If we instead train against a
dataset-derived 141-entry dictionary, the head becomes 163 classes, the pretrained
head is discarded, and recognition has to relearn from scratch. So we keep the base
dictionary order and only append characters the dataset genuinely needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import MAX_TEXT_LENGTH


@dataclass
class Record:
    image: str
    text: str
    domain: str = "unknown"
    script: str = "unknown"
    source_dataset: str = "unknown"
    original_filename: str = ""
    width: int = 0
    height: int = 0


def read_annotations(root: Path, split: str) -> list[tuple[str, str]]:
    path = root / f"{split}_rec.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_dataset.py to build the dataset first."
        )
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "\t" not in line:
            raise ValueError(f"{path}:{number} has no tab separator")
        image, text = line.split("\t", 1)
        records.append((image, text))
    return records


def read_records(root: Path, split: str) -> list[Record]:
    """Read annotations plus the metadata sidecar written by prepare_dataset.py."""
    base = read_annotations(root, split)
    meta_path = root / f"{split}_meta.jsonl"
    if not meta_path.is_file():
        return [Record(image=image, text=text) for image, text in base]

    meta_lines = meta_path.read_text(encoding="utf-8").splitlines()
    if len(meta_lines) != len(base):
        raise ValueError(
            f"{meta_path} has {len(meta_lines)} rows but {split}_rec.txt has {len(base)}; "
            "re-run scripts/prepare_dataset.py"
        )

    records = []
    for (image, text), raw in zip(base, meta_lines):
        entry = json.loads(raw)
        records.append(
            Record(
                image=entry.get("image", image),
                text=text,
                domain=entry.get("domain", "unknown"),
                script=entry.get("script", "unknown"),
                source_dataset=entry.get("source_dataset", "unknown"),
                original_filename=entry.get("original_filename", ""),
                width=int(entry.get("width", 0) or 0),
                height=int(entry.get("height", 0) or 0),
            )
        )
    return records


def load_base_dictionary(base_model_dir: Path) -> list[str]:
    """Read the exact character order the base checkpoint was trained with."""
    inference_yml = base_model_dir / "inference.yml"
    if not inference_yml.is_file():
        raise FileNotFoundError(
            f"{inference_yml} not found. The base inference model is required to build the dictionary."
        )
    config = yaml.safe_load(inference_yml.read_text(encoding="utf-8"))
    entries = (config.get("PostProcess") or {}).get("character_dict")
    if not entries:
        raise ValueError(f"{inference_yml} has no PostProcess.character_dict")
    return list(entries)


def build_dictionary(
    base_entries: list[str], dataset_characters: set[str], use_space_char: bool = True
) -> tuple[list[str], list[str]]:
    """Keep the pretrained order, appending only genuinely new characters.

    The space character is deliberately *not* appended when `use_space_char` is set,
    because PaddleOCR already appends it to the dictionary at load time. Writing it
    into the file as well would add a second space class and shift the head by one.

    Returns (entries, appended). `appended` must be empty for the pretrained CTC head
    to transfer unchanged; a non-empty result is a reported condition rather than a
    silent degradation.
    """
    entries = list(base_entries)
    present = set(entries)
    candidates = dataset_characters - ({" "} if use_space_char else set())
    appended = sorted(character for character in candidates if character not in present)
    entries.extend(appended)
    return entries, appended


def write_dictionary(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for character in entries:
            handle.write(f"{character}\n")


def expected_head_classes(entries: list[str], use_space_char: bool = True) -> dict[str, int]:
    """Class counts PaddleOCR will build, mirroring ppocr.data.imaug.label_ops."""
    characters = len(entries) + (1 if use_space_char else 0)
    return {
        "ctc": 1 + characters,          # CTCLabelEncode prepends "blank"
        "nrtr": 4 + characters,         # NRTRLabelEncode prepends blank/<unk>/<s>/</s>
    }


def pretrained_head_classes(pretrained_path: Path) -> dict[str, int]:
    """Read the true output sizes straight out of the checkpoint tensors."""
    import paddle

    state = paddle.load(str(pretrained_path))
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    ctc = state["head.ctc_head.fc.bias"].shape[0]
    nrtr = state["head.gtc_head.tgt_word_prj.weight"].shape[1]
    return {"ctc": int(ctc), "nrtr": int(nrtr)}


def verify_dictionary(entries: list[str], pretrained_path: Path, use_space_char: bool = True) -> dict:
    """Compare the head the dictionary implies against the head the checkpoint has."""
    built = expected_head_classes(entries, use_space_char)
    actual = pretrained_head_classes(pretrained_path)
    return {
        "dictionary_entries": len(entries),
        "use_space_char": use_space_char,
        "expected": built,
        "pretrained": actual,
        "ctc_matches": built["ctc"] == actual["ctc"],
        "nrtr_matches": built["nrtr"] == actual["nrtr"],
        # The CTC branch drives inference (CTCLabelDecode). The NRTR branch is a
        # training-time auxiliary loss, so a mismatch there is a warning, not a blocker.
        "head_transfers": built["ctc"] == actual["ctc"],
    }


def assert_labels_within_limit(records: list[Record], limit: int = MAX_TEXT_LENGTH) -> None:
    too_long = [record for record in records if len(record.text) > limit]
    if too_long:
        examples = ", ".join(f"{record.image}={record.text!r}" for record in too_long[:5])
        raise ValueError(
            f"{len(too_long)} label(s) exceed max_text_length={limit}: {examples}. "
            "Raise MAX_TEXT_LENGTH in scripts/lib/paths.py if this is expected."
        )
