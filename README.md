# Kurdish OCR Recognition Fine-Tuning & High-Performance Pipeline (PaddleOCR)

An end-to-end, production-ready framework for fine-tuning PaddleOCR recognition models on Kurdish text (**Sorani** Arabic-script and **Kurmanji** Latin-script, handling right-to-left visual ordering, ligatures, digits, English, and Zero-Width Non-Joiner ZWNJ characters).

Built following official **PaddlePaddle / PaddleOCR v5** guidelines with **zero catastrophic forgetting** of base representations.

---

## Key Highlights

- **Foundation Model**: `arabic_PP-OCRv5_mobile_rec` (`SVTR_LCNet` backbone + `PPLCNetV3` + `svtr` neck + `MultiHead` CTC/NRTR).
- **Zero Catastrophic Forgetting**: Preserves all 747 base characters (Arabic, English, Persian, 0-9 digits, Persian/Arabic numerals, punctuation, mathematical symbols). CTC classification head matches the pretrained checkpoint tensor `[749, 120]` byte-for-byte with 0 re-initialization.
- **100% Character Coverage**: Verified against all 162,000+ Kurdish text samples in `qai-ocr-v1-small` with 0 missing characters.
- **RTL Visual Order Handling**: Built-in visual-to-logical Unicode reversal adhering to PaddleOCR's native RTL decoding architecture.
- **High Throughput & Low Latency**: ~80 FPS, ~12 ms per text-line image on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- **Deployable Export**: One-step export to lightweight, frozen Paddle 3.x PIR inference format (`inference.json`, `inference.pdiparams`, `inference.yml`).
- **Pretty Terminal Formatting**: Clean ANSI and tabular terminal reporting with instant CER, WER, latency, and accuracy metrics.

---

## Project Structure

```
qai-paddle-ocr-finetuning/
├── main.py                     # Unified CLI entrypoint (prepare, smoke-test, train, export, infer)
├── Makefile                    # Standard developer automation targets
├── requirements.txt            # Unified Python dependencies
├── configs/
│   ├── kurdish_rec.yml         # Production training configuration (PP-OCRv5)
│   └── arabic_kurdish_dict.txt # 747-character dictionary with RTL trigger
├── src/
│   ├── benchmark/engine.py     # Evaluation and CTC decoding engine
│   ├── data/processor.py       # WebDataset shard reader and dataset converter
│   ├── data/normalizer.py      # Kurdish NFKC and ZWNJ orthography normalizer
│   ├── metrics/distance.py     # Fast Levenshtein distance, CER, WER, Exact Match
│   ├── pipeline/smoke_test.py  # CUDA, VRAM, and gradient graph validation
│   ├── pipeline/pilot_run.py   # Rapid mini-subset convergence runner
│   ├── pipeline/trainer.py     # Full multi-epoch trainer and checkpoint manager
│   ├── pipeline/exporter.py    # Inference model exporter
│   └── utils/visualizer.py     # Pretty ANSI table renderer
├── scripts/
│   ├── setup.ps1               # Automated Windows setup script
│   ├── download_base_models.py # Auto-downloads official base models if missing
│   ├── verify_env.py           # Deep environment and checksum verification
│   └── test_samples.py         # Visual test utility on individual images
└── pretrain_models/            # Base pretrained weights (arabic_PP-OCRv5_mobile_rec)
```

---

## Quick Setup (Windows PC)

### Option 1: Automated 1-Click Setup (Recommended)

Open **PowerShell** in the project directory and run:

```powershell
.\scripts\setup.ps1
```

The script will automatically:
1. Create a Python virtual environment (`.venv`).
2. Install GPU-accelerated PaddlePaddle (`cu126`) and project requirements.
3. Clone and check out the exact pinned commit of PaddleOCR.
4. Verify or download official base models with SHA-256 checksums.
5. Run the complete environment verification.

---

### Option 2: Manual Setup

If you prefer setting up step-by-step:

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install PaddlePaddle GPU for CUDA 12.6
pip install "paddlepaddle-gpu==3.3.1" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 4. Install repository dependencies
pip install -r requirements.txt

# 5. Clone PaddleOCR at pinned commit
git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
git -C PaddleOCR checkout 2661c7c0ef5c613e8f93c6e93b2e052399f0f854

# 6. Verify / download official base models
python scripts/download_base_models.py
```

---

## Dataset Setup

Place your Hugging Face dataset folder directly inside `data/`:

```
qai-paddle-ocr-finetuning/
└── data/
    └── qai-ocr-v1-small/
        ├── train-000000.tar
        ├── train-000001.tar
        ├── ...
        ├── val-000000.tar
        └── test-000000.tar
```

Then extract and format the dataset with:

```powershell
python main.py prepare-data
```

This will automatically extract the WebDataset TAR shards into `data/kurdish_rec/`:
- `117,403` training samples
- `13,981` validation samples
- `30,734` test samples
- Label files: `train_rec.txt`, `val_rec.txt`, `test_rec.txt`

---

## End-to-End Workflow

### 1. Hardware & Gradient Smoke Test
Verify that your GPU, CUDA drivers, VRAM, and gradient backpropagation are functional:

```powershell
python main.py smoke-test
```

### 2. Baseline Model Benchmark
Evaluate the zero-shot baseline model on Kurdish test data to establish the pre-training accuracy:

```powershell
python main.py benchmark-base --max-samples 100
```

### 3. Rapid Pilot Run
Run a fast convergence test (e.g. 50 samples, 1 epoch) to verify that loss decreases, checkpoints save, and auto-evaluation works:

```powershell
python main.py pilot-run --num-samples 50 --max-epochs 1 --batch-size 16
```

### 4. Full Production Training
Fine-tune the model on your dataset with automatic best-checkpoint tracking:

```powershell
# Default uses configs/kurdish_rec.yml (40 epochs, learning rate 0.001)
python main.py train --epochs 40 --gpus 0
```

Checkpoints will be saved to:
`output/production_run/checkpoints/` (`best_accuracy.pdparams`, `latest.pdparams`)

### 5. Checkpoint Leaderboard
Evaluate all generated checkpoints and render a formatted leaderboard:

```powershell
python main.py benchmark-all --checkpoints-dir output/production_run/checkpoints --max-samples 500
```

### 6. Export to Inference Format
Export the best trained checkpoint into deployable inference format:

```powershell
python main.py export --checkpoint output/production_run/checkpoints --output-dir export/kurdish_final
```

Exported files in `export/kurdish_final/`:
- `inference.json` (network graph architecture)
- `inference.pdiparams` (optimized weights)
- `inference.yml` (model configuration)
- `arabic_kurdish_dict.txt` (character vocabulary)

### 7. Run Inference

#### Via CLI:
```powershell
# Recognize Kurdish text on an individual image:
python main.py infer --model-dir export/kurdish_final --image path/to/sample.jpg

# Test recognition on 10 random samples from the test split with ground truth comparison:
python main.py infer --model-dir export/kurdish_final --split test --count 10
```

#### Via Python in your application:
```python
from pathlib import Path
from scripts.lib.recognizer import Recognizer

# Initialize recognizer
model_dir = Path("export/kurdish_final")
recognizer = Recognizer(model_dir=model_dir, batch_size=16, use_gpu=True)

# Run inference on image files
image_paths = [Path("test_sample.png")]
predictions, timing = recognizer.predict_paths(image_paths)

for pred in predictions:
    print(f"Recognized: {pred.text} (Confidence: {pred.score:.4f})")

print(f"Throughput: {timing['images_per_second']:.1f} FPS")
```

---

## Git & Repository Rules

The repository is configured to exclude large generated artifacts and local data:

- `data/` is strictly **excluded** from Git (keep datasets local).
- `output/`, `export/`, and `reports/` are excluded from Git.
- `.venv/` and `PaddleOCR/` checkouts are excluded from Git.

To push your repository cleanly to GitHub:

```powershell
git add .
git commit -m "feat: complete PaddleOCR Kurdish fine-tuning and inference pipeline"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
