#!/usr/bin/env bash
# ==============================================================================
# Linux / Google Colab / WSL Automated Setup Script for Kurdish PaddleOCR
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo " Kurdish OCR Environment Setup (Linux / Colab / WSL)"
echo "========================================================================"

# 1. Virtual Environment (if not inside Google Colab or Docker container)
PYTHON="python3"
if [ ! -d "/content" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
        echo "[*] Creating virtual environment (.venv)..."
        python3 -m venv .venv
    fi
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
    source .venv/bin/activate
fi

# 2. Upgrade pip
echo "[*] Upgrading pip..."
${PYTHON} -m pip install -U pip setuptools wheel --quiet

# 3. Detect Hardware & Install PaddlePaddle
echo "[*] Detecting compute hardware..."
CUDA_VER=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | awk '{print $3}' || true)
fi

if [[ -n "${CUDA_VER}" && $(echo "${CUDA_VER} >= 12.0" | bc -l 2>/dev/null || echo 1) -eq 1 ]]; then
    echo "[*] NVIDIA GPU detected with CUDA ${CUDA_VER} -> Installing paddlepaddle-gpu 3.3.1 (cu126)..."
    ${PYTHON} -m pip install "paddlepaddle-gpu==3.3.1" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --quiet
elif [[ -n "${CUDA_VER}" && $(echo "${CUDA_VER} >= 11.0" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
    echo "[*] NVIDIA GPU detected with CUDA ${CUDA_VER} -> Installing paddlepaddle-gpu 3.3.1 (cu118)..."
    ${PYTHON} -m pip install "paddlepaddle-gpu==3.3.1" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/ --quiet
else
    echo "[*] No CUDA GPU detected -> Installing CPU-only paddlepaddle 3.3.1..."
    ${PYTHON} -m pip install "paddlepaddle==3.3.1" --quiet
fi

# 4. Clone PaddleOCR at pinned commit
PADDLE_COMMIT="2661c7c0ef5c613e8f93c6e93b2e052399f0f854"
PADDLE_REPO="https://github.com/PaddlePaddle/PaddleOCR.git"

if [ -d "PaddleOCR/.git" ]; then
    CURRENT=$(git -C PaddleOCR rev-parse HEAD 2>/dev/null || true)
    if [ "${CURRENT}" = "${PADDLE_COMMIT}" ]; then
        echo "[OK] PaddleOCR already at pinned commit ${PADDLE_COMMIT}"
    else
        echo "[*] Checking out pinned commit ${PADDLE_COMMIT} in PaddleOCR..."
        git -C PaddleOCR fetch --all --tags --quiet
        git -C PaddleOCR checkout "${PADDLE_COMMIT}" --quiet
    fi
else
    echo "[*] Cloning PaddleOCR repository..."
    git clone "${PADDLE_REPO}" PaddleOCR --quiet
    git -C PaddleOCR checkout "${PADDLE_COMMIT}" --quiet
    echo "[OK] PaddleOCR pinned to ${PADDLE_COMMIT}"
fi

# 5. Install unified dependencies
echo "[*] Installing project dependencies (requirements.txt)..."
${PYTHON} -m pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed"

# 6. Verify / download official base models
echo "[*] Verifying base models..."
${PYTHON} scripts/download_base_models.py
echo "[OK] Base models ready"

# 7. Dataset Download & Preparation
echo "[*] Preparing dataset from Hugging Face..."
${PYTHON} scripts/prepare_dataset.py --skip-if-exists
echo "[*] Auditing dataset..."
${PYTHON} scripts/audit_dataset.py
echo "[OK] Dataset ready and audited"

# 8. Run environment verification
echo "========================================================================"
echo " Verifying Environment"
echo "========================================================================"
${PYTHON} scripts/verify_env.py --require-data

echo ""
echo "========================================================================"
echo " Setup Complete & Ready!"
echo "========================================================================"
echo " Run any of the following commands:"
echo "   ${PYTHON} main.py smoke-test"
echo "   ${PYTHON} main.py benchmark-base --max-samples 100"
echo "   ${PYTHON} main.py pilot-run --num-samples 500"
echo "   ${PYTHON} main.py train --epochs 40"
echo "   ${PYTHON} main.py export"
echo "   ${PYTHON} main.py infer --split test --count 5"
echo "========================================================================"
