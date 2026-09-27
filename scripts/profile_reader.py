"""Profile the recognition data pipeline to find what actually limits training speed.

The batch probe showed the GPU idling at 210-345 MHz while `avg_reader_cost` took ~65%
of every step, so the loader is the bottleneck. This runs the *real* SimpleDataSet with
the *real* transforms in a single process and times each stage, so the slow part is
identified instead of guessed at.

    python scripts/profile_reader.py
    python scripts/profile_reader.py --samples 500 --decode-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.paths import PADDLE_ROOT, REC_ROOT, TRAIN_CONFIG
else:  # pragma: no cover
    from .lib.paths import PADDLE_ROOT, REC_ROOT, TRAIN_CONFIG

sys.path.insert(0, str(PADDLE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=400, help="Samples to time")
    parser.add_argument("--offset", type=int, default=0, help="Start row in the label file")
    parser.add_argument("--decode-only", action="store_true",
                        help="Time only cv2.imread, skipping PaddleOCR's transforms")
    return parser.parse_args()


def read_lines(offset: int, samples: int) -> list[str]:
    lines = (REC_ROOT / "train_rec.txt").read_text(encoding="utf-8").splitlines()
    return lines[offset:offset + samples]


def time_decode(paths: list[Path]) -> tuple[float, int]:
    import cv2
    start = time.perf_counter()
    pixels = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            pixels += image.shape[0] * image.shape[1]
    return time.perf_counter() - start, pixels


def time_full_dataset(paths: list[Path], samples: int) -> tuple[float, int]:
    from ppocr.data.simple_dataset import SimpleDataSet
    from ppocr.data import create_operators
    from tools.program import load_config

    # Same dataset object training uses, with the real config, pointed at a small slice.
    config = load_config(str(TRAIN_CONFIG))
    # data_dir is written relative to the PaddleOCR checkout, so make it absolute here.
    config["Global"]["character_dict_path"] = str(TRAIN_CONFIG.parent / "arabic_kurdish_dict.txt")
    config["Train"]["dataset"]["data_dir"] = str(REC_ROOT) + "/"
    config["Train"]["dataset"]["label_file_list"] = [str(REC_ROOT / "_profile_slice.txt")]
    config["Train"]["dataset"]["transforms"] = [
        {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
        {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
        {"RecResizeImg": {"image_shape": [3, 48, 320]}},
        {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
    ]

    class _Logger:
        def info(self, *args, **kwargs):
            pass
        warning = error = debug = info

    dataset = SimpleDataSet(config, "Train", _Logger())

    start = time.perf_counter()
    for index in range(min(samples, len(dataset))):
        dataset[index]
    return time.perf_counter() - start, len(dataset)


def time_real_loader(paths: list[Path], samples: int, batch: int, workers: int,
                     shared_memory: bool) -> tuple[float, int, int]:
    """Time the actual paddle DataLoader, so worker overhead and inter-process transfer
    are measured rather than assumed away by the single-process numbers above."""
    import paddle
    from ppocr.data import build_dataloader
    from tools.program import load_config

    config = load_config(str(TRAIN_CONFIG))
    config["Global"]["character_dict_path"] = str(TRAIN_CONFIG.parent / "arabic_kurdish_dict.txt")
    config["Train"]["dataset"]["data_dir"] = str(REC_ROOT) + "/"
    config["Train"]["dataset"]["label_file_list"] = [str(REC_ROOT / "_profile_slice.txt")]
    # A plain BatchSampler here: we want the loader's cost, not MultiScaleSampler's mixing.
    config["Train"].pop("sampler", None)
    config["Train"]["loader"].update({
        "batch_size_per_card": batch,
        "num_workers": workers,
        "shuffle": False,
        "drop_last": True,
        "use_shared_memory": shared_memory,
    })

    class _Logger:
        def info(self, *args, **kwargs):
            pass
        warning = error = debug = info

    loader = build_dataloader(config, "Train", paddle.CPUPlace(), _Logger())

    start = time.perf_counter()
    seen = 0
    batches = 0
    for batch_data in loader:
        seen += len(batch_data[0])
        batches += 1
        if seen >= samples:
            break
    elapsed = time.perf_counter() - start
    del loader
    return elapsed, seen, batches


def main() -> int:
    args = parse_args()
    lines = read_lines(args.offset, args.samples)
    if not lines:
        print("no samples available", file=sys.stderr)
        return 1
    paths = [REC_ROOT / line.split("\t")[0] for line in lines]

    print(f"profiling {len(paths)} samples (single process, no workers)")
    decode_seconds, _ = time_decode(paths)
    print(f"  cv2.imread only      {decode_seconds:7.3f} s  "
          f"({decode_seconds / len(paths) * 1000:6.2f} ms/img, {len(paths) / decode_seconds:8.1f} img/s)")

    slice_file = REC_ROOT / "_profile_slice.txt"
    slice_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        full_seconds, _ = time_full_dataset(paths, len(paths))
        print(f"  full SimpleDataSet  {full_seconds:7.3f} s  "
              f"({full_seconds / len(paths) * 1000:6.2f} ms/img, {len(paths) / full_seconds:8.1f} img/s)")
        print(f"  transforms add      {full_seconds - decode_seconds:7.3f} s  "
              f"({(full_seconds - decode_seconds) / len(paths) * 1000:6.2f} ms/img, "
              f"{(full_seconds - decode_seconds) / full_seconds * 100:5.1f}% of pipeline)")

        print("\nnow the real paddle DataLoader (workers + inter-process transfer):")
        for batch, workers in ((32, 8), (256, 8)):
            for shared in (True, False):
                elapsed, seen, batches = time_real_loader(
                    paths, len(paths), batch, workers, shared)
                print(f"  batch {batch:>3} workers {workers:>2} shm={str(shared):<5} "
                      f"{elapsed:7.3f} s  {seen / elapsed:8.1f} img/s  "
                      f"({batches} batches, {seen} samples)")
    finally:
        slice_file.unlink(missing_ok=True)

    print("\n  Training measured ~50 img/s on the GPU. If the loader above is far faster,")
    print("  the bottleneck is the training loop, not data loading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
