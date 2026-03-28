#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="nips-specscale"

echo "============================================"
echo " Environment Setup (uv + PyTorch 2.10 + CUDA 12.8)"
echo "============================================"

# --- Install uv if missing ---
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating Python 3.10 venv ..."
    uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
else
    echo "[2/5] Venv exists: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --- Install PyTorch 2.10 + CUDA 12.8 ---
echo "[3/5] Installing PyTorch 2.10.0 + CUDA 12.8 ..."
uv pip install "torch==2.10.0" "torchvision" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128

# --- Install project dependencies (alibaba mirror) ---
echo "[4/5] Installing project dependencies ..."
uv pip install -r "$PROJ_DIR/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match

# --- Optional: flash-attention ---
echo "[5/5] Installing flash-attn (optional) ..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

# --- Verify ---
echo ""
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Run:       bash scripts/run_all_experiments.sh"
