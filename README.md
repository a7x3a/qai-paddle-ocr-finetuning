# Kurdish OCR Recognition Fine-Tuning & High-Performance Pipeline (PaddleOCR)

An end-to-end, production-ready framework for fine-tuning official **PaddleOCR v5** recognition models on Kurdish text (**Sorani** Arabic-script and **Kurmanji** Latin-script, handling right-to-left visual ordering, ligatures, digits, English, and Zero-Width Non-Joiner ZWNJ characters).

Designed for **1-click reproduction across machines** (Windows PC, Linux server, WSL2, and **Google Colab Free Tier T4 GPU**) with zero manual hassle.

---

## Key Highlights

- **Foundation Model**: `arabic_PP-OCRv5_mobile_rec` (`SVTR_LCNet` backbone with `PPLCNetV3` + `svtr` neck + `MultiHead` CTC/NRTR).
- **Official PaddleOCR Git Integration**: Training directly leverages the official PaddleOCR repository (`PaddleOCR/tools/train.py`, `export_model.py`) pinned to stable commit `2661c7c0ef5c613e8f93c6e93b2e052399f0f854`.
- **Zero Catastrophic Forgetting**: Preserves all 747 base characters (Arabic, English, Persian, 0-9 digits, Persian/Arabic numerals, punctuation, mathematical symbols). The CTC head matches the pretrained checkpoint tensor `[749, 120]` byte-for-byte with 0 re-initialization.
- **100% Character Coverage**: Verified against all 162,000+ Kurdish text samples in `a7x3a/qai-ocr-v1-small` with 0 missing characters.
- **Automated Self-Healing Setup**: 1-click installer automatically detects GPU/CPU, downloads official base weights, downloads dataset shards from Hugging Face, extracts splits, and verifies the pipeline.
- **Free Resource & Budget GPU Optimization**:
  - Automatically profiles hardware (VRAM & RAM) to set optimal batch sizes (multiple of 16 for MultiScaleSampler) and worker counts.
  - Automatically enables mixed precision (`amp_level: "O2"`) on GPUs to cut VRAM in half and double throughput (~80 FPS on a 6 GB laptop GPU, ~120+ FPS on Colab T4).
  - Conservative evaluation batch size (<= 96) to prevent mid-training validation OOM.
- **RTL Visual Order Handling**: Built-in visual-to-logical Unicode reversal adhering to PaddleOCR's native RTL decoding architecture (triggered by `arabic` dictionary naming convention).
- **Clean & Minimal Architecture**: Single source of truth in `src/`, unified `requirements.txt`, unified CLI entrypoint `main.py`, and 0 code duplication.

---

## Project Structure

```
qai-paddle-ocr-finetuning/
├── setup.bat                   # 1-Click Windows launcher (bypasses execution policy)
├── main.py                     # Unified CLI entrypoint (setup, prepare, train, export, infer)
├── Makefile                    # Developer automation targets
├── requirements.txt            # Unified Python dependencies
├── configs/
│   ├── kurdish_rec.yml         # Production training configuration (PP-OCRv5)
│   └── arabic_kurdish_dict.txt # 747-character dictionary with RTL trigger
├── src/
│   ├── benchmark/engine.py     # Evaluation and CTC decoding engine
│   ├── data/processor.py       # WebDataset shard reader & HuggingFace auto-downloader
│   ├── data/normalizer.py      # Kurdish NFKC and ZWNJ orthography normalizer
│   ├── metrics/distance.py     # Fast Levenshtein distance, CER, WER, Exact Match
│   ├── metrics/evaluator.py    # Metric aggregation & latency profiling
│   ├── pipeline/smoke_test.py  # CUDA, VRAM, and gradient graph validation
│   ├── pipeline/pilot_run.py   # Rapid mini-subset convergence runner
│   ├── pipeline/trainer.py     # Subprocess runner for PaddleOCR official train.py
│   ├── pipeline/exporter.py    # Inference model exporter
│   ├── utils/hardware.py       # Dynamic GPU/CPU hardware profiler & batch sizing
│   ├── utils/paths.py          # Centralized repository filesystem constants
│   ├── utils/config_injector.py# Dynamic YAML config modifier & validator
│   └── utils/visualizer.py     # Pretty ANSI table renderer
├── scripts/
│   ├── setup.ps1               # Automated PowerShell setup script (Windows)
│   ├── setup.sh                # Automated Bash setup script (Linux / Google Colab)
│   ├── download_base_models.py # Downloads & verifies official PP-OCRv5 weights
│   ├── prepare_dataset.py      # Downloads & extracts WebDataset shards into data/
│   ├── audit_dataset.py        # Strict integrity check of prepared dataset
│   └── verify_env.py           # Deep environment and checksum verification
├── assets/base_rec_inference/  # Official base inference model
└── pretrain_models/            # Official pretrained weights (.pdparams)
```

---

## Quick Setup (1-Click Installation)

Clone this repository onto your machine:
```bash
git clone <your-repository-url>
cd qai-paddle-ocr-finetuning
```

Choose your platform below:

### Option A: Windows PC (1-Click)

Simply run the batch script in Command Prompt or PowerShell:
```cmd
setup.bat
```
*(Or in PowerShell: `powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1`)*

**What the setup script does completely automatically:**
1. Creates a Python virtual environment (`.venv`).
2. Detects your GPU (NVIDIA CUDA 12.x, 11.x, or CPU) and installs the matching `paddlepaddle-gpu` or `paddlepaddle` build.
3. Clones the official PaddleOCR repository and checks out the exact pinned commit.
4. Installs all project dependencies from `requirements.txt`.
5. Verifies and downloads official `arabic_PP-OCRv5_mobile_rec` base weights with SHA-256 integrity checks.
6. Downloads the Kurdish OCR dataset (`a7x3a/qai-ocr-v1-small`) from Hugging Face and extracts it into `data/kurdish_rec/`.
7. Audits the dataset and verifies the entire environment end-to-end.

---

### Option B: Linux / Google Colab (Free T4 GPU) / WSL

In terminal or a Colab cell, run:
```bash
bash scripts/setup.sh
```

#### Running on Google Colab (Free Tier):
1. In Google Colab, change runtime type to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
2. Run in a cell:
   ```bash
   !git clone <your-repository-url>
   %cd qai-paddle-ocr-finetuning
   !bash scripts/setup.sh
   ```
3. Run training:
   ```bash
   !python main.py train --epochs 40
   ```

---

### Option C: Universal Python Command (Cross-Platform)

If you already have Python and pip active:
```bash
python main.py setup
```

---

### Option D: Manual Step-by-Step Installation

If you prefer to set up manually step-by-step:

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 2. Upgrade pip
python -m pip install -U pip setuptools wheel

# 3. Install PaddlePaddle matching your hardware:
# For CUDA 12.x:
pip install "paddlepaddle-gpu==3.3.1" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
# For CUDA 11.x:
pip install "paddlepaddle-gpu==3.3.1" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
# For CPU only:
pip install "paddlepaddle==3.3.1"

# 4. Clone PaddleOCR at pinned commit
git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
git -C PaddleOCR checkout 2661c7c0ef5c613e8f93c6e93b2e052399f0f854

# 5. Install unified dependencies
pip install -r requirements.txt

# 6. Download official PP-OCRv5 base models
python scripts/download_base_models.py

# 7. Download and prepare dataset from Hugging Face
python main.py prepare-data

# 8. Verify the environment
python scripts/verify_env.py --require-data
```

---

## Dataset Pipeline

The dataset is hosted on Hugging Face: [a7x3a/qai-ocr-v1-small](https://huggingface.co/datasets/a7x3a/qai-ocr-v1-small).

Running setup or `python main.py prepare-data` will automatically download the WebDataset TAR shards and extract them into `data/kurdish_rec/`:
- `117,403` training samples (`train_rec.txt`)
- `13,981` validation samples (`val_rec.txt`)
- `30,734` test samples (`test_rec.txt`)
- Row-aligned metadata: `train_meta.jsonl`, `val_meta.jsonl`, `test_meta.jsonl`
- Verification of 100% dictionary character coverage against `configs/arabic_kurdish_dict.txt`.

*(If the dataset has already been extracted, `main.py prepare-data` detects it and finishes instantly).*

---

## End-to-End Workflow

The CLI entrypoint `main.py` provides a unified interface for the entire lifecycle:

### 1. Hardware & Gradient Smoke Test
Verify that your GPU, CUDA runtime, VRAM allocation, and gradient backpropagation are functional:
```bash
python main.py smoke-test
```

### 2. Baseline Model Benchmark
Evaluate the zero-shot base model on Kurdish validation data to measure pre-training accuracy:
```bash
python main.py benchmark-base --max-samples 100
```

### 3. Rapid Pilot Run
Run a fast convergence test (e.g. 500 samples, 2 epochs) to verify that loss decreases, checkpoints save, and auto-evaluation works:
```bash
python main.py pilot-run --num-samples 500 --max-epochs 2
```

### 4. Full Production Fine-Tuning
Fine-tune the model on your dataset with automatic best-checkpoint tracking:
```bash
# Automatically detects VRAM and scales batch size (or specify --batch-size)
python main.py train --epochs 40 --gpus 0
```

Training features:
- **Official PaddleOCR Engine**: Executes `PaddleOCR/tools/train.py` with runtime configuration.
- **Mixed Precision (`O2`)**: Automatically enabled on CUDA GPUs for 2x faster execution and 50% lower VRAM.
- **Dynamic Batch Sizing**: Auto-calculated based on available GPU memory (e.g. 384 on 6 GB+, 128 on 4 GB, 16 on CPU).
- **Two Frozen Backbone Epochs**: CTC/NRTR recognition heads adapt to Kurdish script before backbone weights unfreeze.
- **Checkpoints Saved To**: `output/production_run/checkpoints/` (`best_accuracy.pdparams`, `latest.pdparams`).

### 5. Checkpoint Leaderboard
Evaluate all generated checkpoints and render a formatted leaderboard:
```bash
python main.py benchmark-all --checkpoints-dir output/production_run/checkpoints --max-samples 500
```

### 6. Export to Inference Format
Export the trained checkpoint into lightweight, deployable Paddle PIR inference format:
```bash
python main.py export --checkpoint output/production_run/checkpoints --output-dir export/kurdish_final
```

Exported artifacts in `export/kurdish_final/`:
- `inference.json` (network computational graph)
- `inference.pdiparams` (optimized weights)
- `inference.yml` (model configuration)
- `arabic_kurdish_dict.txt` (747-character dictionary)

### 7. Run Inference

#### Via CLI:
```bash
# Test recognition on 10 random samples from the test split with ground truth diff:
python main.py infer --model-dir export/kurdish_final --split test --count 10

# Recognize Kurdish text on an individual image:
python main.py infer --model-dir export/kurdish_final --image path/to/sample.jpg
```

#### Via Python in your application:
```python
from pathlib import Path
from scripts.lib.recognizer import Recognizer

# Initialize recognizer (runs on GPU or CPU)
model_dir = Path("export/kurdish_final")
recognizer = Recognizer(model_dir=model_dir, batch_size=16, use_gpu=True)

# Run inference on image files
image_paths = [Path("sample1.jpg"), Path("sample2.png")]
predictions, timing = recognizer.predict_paths(image_paths)

for pred in predictions:
    print(f"Recognized: {pred.text} (Confidence: {pred.score:.4f})")

print(f"Throughput: {timing['images_per_second']:.1f} FPS")
```

---

## Free Resources & Optimization Guide

| Environment | GPU / Memory | Recommended Batch Size | AMP Level | Workers | Expected Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Google Colab (Free)** | NVIDIA T4 (15 GB) | `256` - `384` | `O2` (FP16) | `2` - `4` | ~120 FPS |
| **Laptop GPU (e.g. RTX 4050/3060)** | 6 GB VRAM / 16 GB RAM | `256` - `384` | `O2` (FP16) | `2` | ~80 FPS |
| **Budget GPU (e.g. GTX 1650)** | 4 GB VRAM / 8 GB RAM | `64` - `128` | `O2` (FP16) | `2` | ~45 FPS |
| **CPU Only (PC / Mac)** | 4-8 Cores / 8-16 GB RAM | `16` - `32` | `O0` (FP32) | `2` | ~8-15 FPS |

### Memory Management & Anti-OOM Protections:
1. **MultiScaleSampler Factor**: In PaddleOCR, `first_bs` must be a multiple of 16 (divided by 8 and 16). The hardware profiler guarantees batch sizes adhere to this constraint.
2. **Evaluation Batch Capping**: Training reserves CUDA memory pools; running a 384-wide evaluation batch during validation would spike memory and cause OOM. The pipeline caps eval batch size to `min(96, batch_size)`.
3. **RAM Worker Budgeting**: Dataloader workers buffer whole batches into host RAM. The hardware profiler budgets ~1.5 GB per worker against available free physical RAM.

---

## Troubleshooting & FAQ

### 1. PowerShell Script Execution Disabled on Windows
If you see `File scripts\setup.ps1 cannot be loaded because running scripts is disabled on this system`:
- **Solution**: Run `setup.bat` instead, or run:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
  ```

### 2. cuDNN Version Warning
If you see:
```text
The installed Paddle is compiled with CUDNN 9.9, but CUDNN version in your machine is 9.5
```
- **Explanation**: This is an upstream PaddlePaddle 3.3.1 packaging note. `paddlepaddle-gpu==3.3.1` wheel pins `nvidia-cudnn-cu12==9.5.1.17`. This warning is cosmetic; training, gradients, and inference are verified 100% correct.

### 3. Out-Of-Memory (OOM) on Small GPUs
- If your GPU has <= 4 GB VRAM, pass a custom batch size:
  ```bash
  python main.py train --batch-size 64
  ```

---

## Git & Repository Hygiene

The repository is configured to keep Git clean:
- `data/` is strictly excluded from Git (datasets remain local).
- `output/`, `export/`, `reports/`, `.venv/`, and `PaddleOCR/` are excluded from Git.

To push changes:
```bash
git add .
git commit -m "refactor: unified pipeline, automated self-healing setup, and zero-forgetting Kurdish PP-OCRv5"
git push
```
