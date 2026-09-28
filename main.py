#!/usr/bin/env python3
"""Unified CLI Entrypoint for PaddleOCR Fine-Tuning, Benchmarking, and MLOps Pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stream reconfiguration on Windows to prevent UnicodeEncodeError: 'charmap'
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Virtual Environment & Dependency Guardrail:
# If running with system Python while a local .venv exists, instruct user to activate it and exit cleanly.
_is_setup = len(sys.argv) > 1 and sys.argv[1] == "setup"
_venv_path = PROJECT_ROOT / ".venv"
if (sys.prefix == sys.base_prefix) and _venv_path.is_dir() and not _is_setup:
    _activate_cmd = r".\.venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(
        f"\n[ALERT] You are running with the system Python interpreter: {sys.executable}\n"
        f"A local virtual environment was detected at: {_venv_path}\n"
        f"Please activate the virtual environment before running pipeline commands:\n\n"
        f"    {_activate_cmd}\n\n"
        f"Then re-run: python main.py {' '.join(sys.argv[1:])}\n",
        file=sys.stderr,
    )
    sys.exit(1)

# Also ensure PaddleOCR repository directory is accessible
if (PROJECT_ROOT / "PaddleOCR").is_dir() and str(PROJECT_ROOT / "PaddleOCR") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PaddleOCR"))

from src.benchmark.engine import BenchmarkEngine
from src.data.processor import DatasetProcessor
from src.pipeline.exporter import ModelExporter
from src.pipeline.pilot_run import PilotRunner
from src.pipeline.smoke_test import SmokeTester
from src.pipeline.trainer import PaddleTrainer
from src.utils.config_injector import PaddleConfigInjector
from src.utils.logger import setup_logger
from src.utils.visualizer import Visualizer

logger = setup_logger("CLI")


def _resolve_default_path(*candidates: str) -> str:
    """Return the first candidate path that exists on disk, or the first candidate as fallback."""
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0] if candidates else ""


def handle_setup(args: argparse.Namespace) -> None:
    """Execute automated environment setup across Windows, Linux, and Google Colab."""
    import subprocess
    from src.utils.hardware import get_hardware_profile
    from src.utils.paths import PADDLE_COMMIT, PADDLE_REPO, PADDLE_ROOT, PROJECT_ROOT

    logger.info("Executing automated environment setup...")
    profile = get_hardware_profile()
    logger.info(f"Compute Hardware: {profile.gpu_name} ({profile.gpu_memory_mb} MB VRAM, {profile.cpu_cores} CPU cores)")

    # 0. Hardware-aware PaddlePaddle installation if missing or requested
    try:
        import paddle
        has_paddle = True
    except ImportError:
        has_paddle = False

    if not has_paddle or getattr(args, "install_paddle", False):
        import re
        logger.info("Resolving optimal PaddlePaddle wheel for detected architecture...")
        gpu_name = profile.gpu_name
        is_blackwell = bool(re.search(r"RTX 50|Blackwell", gpu_name, re.IGNORECASE))
        no_gpu = getattr(args, "no_gpu", False)

        installed = False
        if profile.has_gpu and not no_gpu:
            if is_blackwell:
                logger.info("Blackwell / RTX 50-Series detected -> Installing paddlepaddle-gpu 3.4.0 (cu129)...")
                ret = subprocess.run([
                    sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.4.0",
                    "--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
                    "--timeout", "120", "--retries", "5",
                ]).returncode
                if ret == 0:
                    installed = True

            if not installed:
                logger.info("Installing paddlepaddle-gpu 3.3.1 (cu126)...")
                ret = subprocess.run([
                    sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.3.1",
                    "--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
                    "--timeout", "120", "--retries", "5",
                ]).returncode
                if ret == 0:
                    installed = True
                else:
                    logger.info("cu126 failed; retrying with cu118...")
                    ret = subprocess.run([
                        sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==3.3.1",
                        "--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
                        "--timeout", "120", "--retries", "5",
                    ]).returncode
                    if ret == 0:
                        installed = True

        if not installed:
            logger.info("Installing CPU-only paddlepaddle 3.3.1...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "paddlepaddle==3.3.1",
                "--timeout", "120", "--retries", "5",
            ], check=True)

    # 1. Clone / checkout pinned PaddleOCR
    if not (PADDLE_ROOT / ".git").is_dir():
        logger.info(f"Cloning PaddleOCR repository from {PADDLE_REPO}...")
        subprocess.run(["git", "clone", PADDLE_REPO, str(PADDLE_ROOT)], check=True)
        subprocess.run(["git", "-C", str(PADDLE_ROOT), "checkout", PADDLE_COMMIT], check=True)
        logger.info(f"PaddleOCR checked out at pinned commit: {PADDLE_COMMIT}")
    else:
        logger.info(f"PaddleOCR repository already present at: {PADDLE_ROOT}")

    # 2. Base models
    logger.info("Verifying / downloading official base models...")
    from scripts.download_base_models import ensure_base_models
    ensure_base_models()

    # 3. Dataset preparation
    if not args.skip_dataset:
        logger.info("Verifying / downloading and extracting Kurdish dataset from Hugging Face...")
        processor = DatasetProcessor()
        processor.process_webdataset(
            input_dir="data/qai-ocr-v1-small",
            output_dir="./data/kurdish_rec",
            max_text_length=32,
            overwrite=args.force_data,
        )

    # 4. Verify environment
    logger.info("Running deep environment verification...")
    verify_args = [sys.executable, "scripts/verify_env.py"]
    if not args.skip_dataset:
        verify_args.append("--require-data")
    subprocess.run(verify_args, check=True)
    logger.info("Environment setup completed successfully!")


def handle_prepare_data(args: argparse.Namespace) -> None:
    """Execute dataset transformation, WebDataset shard extraction, or polygon perspective cropping."""
    logger.info("Starting dataset preprocessing pipeline...")
    processor = DatasetProcessor(random_seed=args.seed)

    if args.jsonl_path:
        train_p, val_p, dict_p = processor.process(
            jsonl_path=args.jsonl_path,
            image_root=args.image_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
        )
    else:
        # Default or specified WebDataset directory
        dataset_dir = args.dataset_dir or "data/qai-ocr-v1-small"
        logger.info(f"Processing WebDataset shards from: {dataset_dir}")
        train_p, val_p, dict_p = processor.process_webdataset(
            input_dir=dataset_dir,
            output_dir=args.output_dir,
            max_text_length=args.max_text_length,
            limit=args.limit,
            overwrite=args.overwrite,
        )

    logger.info("Dataset preparation complete:")
    logger.info(f"  - Train label file: {train_p}")
    logger.info(f"  - Validation label file: {val_p}")
    logger.info(f"  - Character vocabulary: {dict_p}")


def handle_export(args: argparse.Namespace) -> None:
    """Export trained checkpoint to deployable PaddleOCR inference format."""
    logger.info("Exporting model checkpoint to inference format...")
    exporter = ModelExporter()
    out_dir = exporter.export(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        dict_path=args.dict_path,
    )
    logger.info(f"Model exported successfully to: {out_dir}")


def handle_infer(args: argparse.Namespace) -> None:
    """Execute text recognition inference on image(s) or benchmark splits using an exported model."""
    logger.info("Executing text recognition inference...")
    from scripts.lib.recognizer import Recognizer
    from scripts.lib.metrics import MetricSuite

    model_dir = Path(args.model_dir).resolve()
    if not (model_dir / "inference.json").is_file() and not (model_dir / "inference.pdmodel").is_file():
        logger.error(f"Inference model files not found in: {model_dir}. Please run 'python main.py export' first.")
        sys.exit(1)

    recognizer = Recognizer(model_dir, batch_size=args.batch_size, use_gpu=not args.no_gpu)
    samples: list[tuple[Path, str]] = []

    if args.image:
        target = Path(args.image).resolve()
        if target.is_file():
            samples.append((target, ""))
        elif target.is_dir():
            for p in sorted(target.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    samples.append((p, ""))
        else:
            logger.error(f"Target path does not exist: {target}")
            sys.exit(1)
    elif args.split:
        label_file = Path(f"data/kurdish_rec/{args.split}_rec.txt").resolve()
        rec_root = label_file.parent
        if not label_file.is_file():
            logger.error(f"Split label file not found: {label_file}")
            sys.exit(1)
        with open(label_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= args.count:
                    break
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    samples.append((rec_root / parts[0], parts[1]))
    else:
        logger.error("Please provide either --image <path> or --split <val|test>")
        sys.exit(1)

    if not samples:
        logger.warning("No images available for inference.")
        return

    logger.info(f"Running inference on {len(samples)} image(s) using {model_dir.name}...")
    predictions, timing = recognizer.predict_paths([p for p, _ in samples])

    has_ground_truth = any(exp for _, exp in samples)
    suite = MetricSuite() if has_ground_truth else None
    rows = []

    for (img_path, expected), pred in zip(samples, predictions):
        status = "OK" if (expected and pred.text == expected) else ("DIFF" if expected else "-")
        if suite and expected:
            suite.add(expected, pred.text, pred.score)
        row = {
            "image": img_path.name,
            "prediction": pred.text,
            "confidence": f"{pred.score:.4f}",
        }
        if has_ground_truth:
            row["expected"] = expected
            row["match"] = status
        rows.append(row)

    print("\n" + Visualizer.render_predictions(rows) + "\n")
    logger.info(f"Throughput: {timing['images_per_second']:.1f} img/s | Avg Latency: {1000.0 / max(timing['images_per_second'], 1e-5):.2f} ms")
    if suite and any(exp for _, exp in samples):
        summary = suite.overall.summary()
        logger.info(f"Exact Match Accuracy: {summary['exact_match_accuracy'] * 100:.2f}% | Corpus CER: {summary['cer'] * 100:.2f}%")



def handle_smoke_test(args: argparse.Namespace) -> None:
    """Execute pre-training environment, CUDA hardware, and gradient graph validation."""
    logger.info("Executing pre-training smoke test...")
    tester = SmokeTester(use_gpu=not args.no_gpu)
    tester.run_all(
        config_path=args.config,
        dict_path=args.dict_path,
        label_path=args.train_label,
    )


def handle_benchmark_base(args: argparse.Namespace) -> None:
    """Benchmark un-finetuned baseline foundation model to establish zero-shot Kurdish accuracy."""
    logger.info("Evaluating baseline foundation model...")
    engine = BenchmarkEngine(
        config_path=args.config,
        dict_path=args.dict_path,
        use_gpu=not args.no_gpu,
    )
    res = engine.run_benchmark(
        model_path=args.model_path,
        val_label_path=args.val_label,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_report_path=args.output_report,
    )
    table_data = [
        {
            "model": "Base Pretrained Model",
            "micro_cer": res.micro_cer,
            "micro_wer": res.micro_wer,
            "accuracy": res.accuracy,
            "avg_latency_ms": res.avg_latency_ms,
            "throughput_fps": res.throughput_fps,
        }
    ]
    print("\n" + Visualizer.render_table(table_data) + "\n")


def handle_pilot_run(args: argparse.Namespace) -> None:
    """Execute rapid convergence pilot run on a mini subset."""
    logger.info("Starting pilot convergence run...")
    runner = PilotRunner()
    result = runner.run_pilot(
        template_config_path=args.config,
        character_dict_path=args.dict_path,
        train_label_path=args.train_label,
        val_label_path=args.val_label,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )
    logger.info(f"Pilot run successfully concluded: {result}")


def handle_train(args: argparse.Namespace) -> None:
    """Execute full fine-tuning training and auto-benchmark generated checkpoints."""
    logger.info("Preparing production fine-tuning run...")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    if batch_size is None:
        from src.utils.hardware import get_hardware_profile
        profile = get_hardware_profile()
        batch_size = profile.recommended_train_batch
        logger.info(f"Auto-selected hardware-optimized batch size: {batch_size} (Device: {profile.gpu_name})")

    injector = PaddleConfigInjector(args.config)
    injector.inject_runtime_paths(
        character_dict_path=args.dict_path,
        train_label_path=args.train_label,
        val_label_path=args.val_label,
        save_model_dir=output_dir / "checkpoints",
        pretrained_model_path=args.pretrained_model,
        batch_size=batch_size,
        epoch_num=args.epochs,
        learning_rate=args.lr,
    )
    runtime_config_path = output_dir / "runtime_config.yml"
    injector.write_runtime_config(runtime_config_path)

    trainer = PaddleTrainer()
    trainer.run_training(
        config_path=runtime_config_path,
        val_label_path=args.val_label,
        dict_path=args.dict_path,
        save_model_dir=output_dir / "checkpoints",
        gpus=args.gpus,
        auto_benchmark=True,
    )


def handle_benchmark_all(args: argparse.Namespace) -> None:
    """Discover and benchmark all checkpoints in the given directory."""
    logger.info(f"Scanning for checkpoints in: {args.checkpoints_dir}")
    trainer = PaddleTrainer()
    trainer.evaluate_all_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        config_path=args.config,
        val_label_path=args.val_label,
        dict_path=args.dict_path,
        max_samples=args.max_samples,
    )


def main() -> None:
    """Parse CLI arguments and dispatch subcommands with intelligent local auto-discovery."""
    parser = argparse.ArgumentParser(
        prog="finetune-paddle-ocr",
        description="Dynamic End-to-End Fine-Tuning Framework for PaddleOCR Recognition Models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    default_dict = _resolve_default_path(
        "configs/arabic_kurdish_dict.txt",
        "data/processed/dictionary.txt",
    )
    default_train = _resolve_default_path(
        "data/kurdish_rec/train_rec.txt",
        "data/processed/train_label.txt",
    )
    default_val = _resolve_default_path(
        "data/kurdish_rec/val_rec.txt",
        "data/processed/val_label.txt",
    )
    default_train_config = _resolve_default_path(
        "configs/kurdish_rec.yml",
        "configs/template_rec_v4.yml",
    )
    default_bench_config = _resolve_default_path(
        "configs/kurdish_rec.yml",
        "configs/template_rec_v4.yml",
    )
    default_model = _resolve_default_path(
        "pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained.pdparams",
        "pretrain_models/arabic_PP-OCRv5_mobile_rec_pretrained",
    )
    default_infer_model = _resolve_default_path(
        "export/kurdish_final",
        "export/pilot_test",
        "assets/base_rec_inference",
    )

    # 0. setup
    p_setup = subparsers.add_parser("setup", help="Automated setup of PaddleOCR, dependencies, base models & dataset")
    p_setup.add_argument("--skip-dataset", action="store_true", help="Skip dataset download/preparation")
    p_setup.add_argument("--force-data", action="store_true", help="Force re-extraction of dataset")
    p_setup.add_argument("--install-paddle", action="store_true", help="Install or update architecture-optimized PaddlePaddle")
    p_setup.add_argument("--no-gpu", action="store_true", help="Force CPU-only installation")
    p_setup.set_defaults(func=handle_setup)

    # 1. prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Extract WebDataset shards or parse JSONL annotations")
    p_prep.add_argument("--dataset-dir", "--input", dest="dataset_dir", type=str, default="data/qai-ocr-v1-small", help="Path to input WebDataset shard root")
    p_prep.add_argument("--jsonl-path", type=str, default=None, help="Optional path to input .jsonl file (for polygon cropping)")
    p_prep.add_argument("--image-dir", type=str, default=None, help="Root directory containing input images (for JSONL)")
    p_prep.add_argument("--output-dir", type=str, default="./data/kurdish_rec", help="Destination folder for processed dataset")
    p_prep.add_argument("--max-text-length", type=int, default=32, help="Maximum label character length (default: 32)")
    p_prep.add_argument("--limit", type=int, default=0, help="Sample limit per split (0 for complete dataset)")
    p_prep.add_argument("--overwrite", action="store_true", help="Force re-extraction of dataset")
    p_prep.add_argument("--val-ratio", type=float, default=0.1, help="Validation split proportion (default: 0.1)")
    p_prep.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed (default: 42)")
    p_prep.set_defaults(func=handle_prepare_data)

    # 2. smoke-test
    p_smoke = subparsers.add_parser("smoke-test", help="Verify hardware, VRAM, and gradient pass")
    p_smoke.add_argument("--config", type=str, default=default_train_config, help="Path to template YAML config")
    p_smoke.add_argument("--dict-path", type=str, default=default_dict, help="Path to character dictionary")
    p_smoke.add_argument("--train-label", type=str, default=default_train, help="Path to train label file")
    p_smoke.add_argument("--no-gpu", action="store_true", help="Force CPU validation only")
    p_smoke.set_defaults(func=handle_smoke_test)

    # 3. benchmark-base
    p_bench_base = subparsers.add_parser("benchmark-base", help="Evaluate un-finetuned foundation model")
    p_bench_base.add_argument("--model-path", type=str, default=default_model, help="Path to base pretrained model or directory")
    p_bench_base.add_argument("--val-label", type=str, default=default_val, help="Path to validation label file")
    p_bench_base.add_argument("--dict-path", type=str, default=default_dict, help="Path to character dictionary")
    p_bench_base.add_argument("--config", type=str, default=default_bench_config, help="Model config YAML")
    p_bench_base.add_argument("--output-report", type=str, default="./output/base_benchmark_report.json", help="Report destination")
    p_bench_base.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    p_bench_base.add_argument("--max-samples", type=int, default=None, help="Limit maximum samples to benchmark (e.g. 100 for fast eval)")
    p_bench_base.add_argument("--no-gpu", action="store_true", help="Run benchmark on CPU")
    p_bench_base.set_defaults(func=handle_benchmark_base)

    # 4. pilot-run
    p_pilot = subparsers.add_parser("pilot-run", help="Run rapid convergence test on mini subset")
    p_pilot.add_argument("--config", type=str, default=default_train_config, help="Template config")
    p_pilot.add_argument("--pretrained-model", type=str, default=default_model, help="Base pretrained weights (.pdparams)")
    p_pilot.add_argument("--dict-path", type=str, default=default_dict, help="Dictionary path")
    p_pilot.add_argument("--train-label", type=str, default=default_train, help="Train labels")
    p_pilot.add_argument("--val-label", type=str, default=default_val, help="Val labels")
    p_pilot.add_argument("--output-dir", type=str, default="./output/pilot", help="Output directory")
    p_pilot.add_argument("--num-samples", type=int, default=500, help="Number of pilot subset samples")
    p_pilot.add_argument("--max-epochs", type=int, default=2, help="Number of mini-epochs")
    p_pilot.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_pilot.set_defaults(func=handle_pilot_run)

    # 5. train
    p_train = subparsers.add_parser("train", help="Run production fine-tuning on full dataset")
    p_train.add_argument("--config", type=str, default=default_train_config, help="Template config")
    p_train.add_argument("--pretrained-model", type=str, default=default_model, help="Base pretrained weights (.pdparams)")
    p_train.add_argument("--dict-path", type=str, default=default_dict, help="Dictionary path")
    p_train.add_argument("--train-label", type=str, default=default_train, help="Train labels")
    p_train.add_argument("--val-label", type=str, default=default_val, help="Val labels")
    p_train.add_argument("--output-dir", type=str, default="./output/production_run", help="Output directory")
    p_train.add_argument("--epochs", type=int, default=40, help="Total training epochs (default: 40)")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size (None for auto-detection)")
    p_train.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    p_train.add_argument("--gpus", type=str, default="0", help="GPU indices (e.g. '0' or '0,1')")
    p_train.set_defaults(func=handle_train)

    # 6. export
    p_export = subparsers.add_parser("export", help="Export trained checkpoint to deployable inference model")
    p_export.add_argument("--checkpoint", type=str, default="./output/production_run/checkpoints", help="Checkpoint path or directory")
    p_export.add_argument("--config", type=str, default=default_train_config, help="Model config YAML")
    p_export.add_argument("--output-dir", type=str, default="./export/kurdish_final", help="Destination export directory")
    p_export.add_argument("--dict-path", type=str, default=default_dict, help="Character dictionary path")
    p_export.set_defaults(func=handle_export)

    # 7. benchmark-all
    p_bench_all = subparsers.add_parser("benchmark-all", help="Evaluate all saved checkpoints in a directory")
    p_bench_all.add_argument("--checkpoints-dir", type=str, default="./output/production_run/checkpoints", help="Directory containing .pdparams")
    p_bench_all.add_argument("--config", type=str, default=default_train_config, help="Model config")
    p_bench_all.add_argument("--val-label", type=str, default=default_val, help="Val labels")
    p_bench_all.add_argument("--dict-path", type=str, default=default_dict, help="Dictionary path")
    p_bench_all.add_argument("--max-samples", type=int, default=None, help="Limit samples to benchmark (e.g. 100 for fast eval)")
    p_bench_all.set_defaults(func=handle_benchmark_all)

    # 8. infer
    p_infer = subparsers.add_parser("infer", help="Run OCR text recognition on images or test splits")
    p_infer.add_argument("--model-dir", type=str, default=default_infer_model, help="Exported model directory")
    p_infer.add_argument("--image", type=str, default=None, help="Path to single image or folder of images")
    p_infer.add_argument("--split", choices=["train", "val", "test"], default=None, help="Benchmark split to run recognition on")
    p_infer.add_argument("--count", type=int, default=10, help="Number of split images to test (default: 10)")
    p_infer.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    p_infer.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    p_infer.set_defaults(func=handle_infer)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
