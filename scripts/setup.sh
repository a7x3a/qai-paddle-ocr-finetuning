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

# 2b. Check Python version compatibility (PaddlePaddle requires 64-bit Python 3.9 - 3.12)
PY_VER=$(${PYTHON} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(${PYTHON} -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(${PYTHON} -c "import sys; print(sys.version_info.minor)")
if [ "${PY_MAJOR}" -ne 3 ] || [ "${PY_MINOR}" -lt 9 ] || [ "${PY_MINOR}" -gt 12 ]; then
    echo "[!] WARNING: PaddlePaddle 3.3.1 officially supports Python 3.9 - 3.12 (current is Python ${PY_VER})."
    echo "[!] If installation fails with 'No matching distribution found', please use Python 3.11 or 3.12."
fi

# 3. Detect Hardware & Install PaddlePaddle
echo "[*] Detecting compute hardware..."
CUDA_MAJOR=""
CUDA_VER=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9]*\.[0-9]*" | head -n1 | awk '{print $3}' || true)
    if [ -n "${CUDA_VER}" ]; then
        CUDA_MAJOR=$(echo "${CUDA_VER}" | cut -d. -f1)
    fi
fi

if [ -n "${CUDA_MAJOR}" ] && [ "${CUDA_MAJOR}" -ge 12 ] 2>/dev/null; then
    echo "[*] NVIDIA GPU detected with CUDA ${CUDA_VER} -> Installing paddlepaddle-gpu 3.3.1 (cu126)..."
    ${PYTHON} -m pip install "paddlepaddle-gpu==3.3.1" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --quiet
elif [ -n "${CUDA_MAJOR}" ] && [ "${CUDA_MAJOR}" -ge 11 ] 2>/dev/null; then
    echo "[*] NVIDIA GPU detected with CUDA ${CUDA_VER} -> Installing paddlepaddle-gpu 3.3.1 (cu118)..."
    ${PYTHON} -m pip install "paddlepaddle-gpu==3.3.1" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/ --quiet
elif command -v nvidia-smi &> /dev/null; then
    echo "[*] NVIDIA GPU detected -> Installing paddlepaddle-gpu 3.3.1 (cu126)..."
    ${PYTHON} -m pip install "paddlepaddle-gpu==3.3.1" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --quiet
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
