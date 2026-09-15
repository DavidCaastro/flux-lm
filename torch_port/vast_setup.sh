#!/bin/bash
# ============================================================
# Flux LM — Vast.ai instance setup script
# Target: RTX 4090 (sm_89) on NVIDIA NGC PyTorch container
# Run this ONCE after SSH into your Vast.ai instance
# ============================================================
set -e

echo "========================================"
echo "  Flux LM — Vast.ai Setup"
echo "========================================"

# ── 0. Verify CUDA toolchain (required for JIT kernel compilation) ──
echo "[0/5] Verifying CUDA toolchain..."
if ! command -v nvcc &>/dev/null; then
    echo "  ERROR: nvcc not found. JIT CUDA kernels will NOT compile."
    echo "  Use an NVIDIA devel image (e.g. NGC PyTorch container)."
    exit 1
fi
echo "  nvcc: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
echo "  gcc:  $(gcc --version | head -1)"

if ! command -v ninja &>/dev/null; then
    echo "  ninja not found, installing..."
    pip3 install ninja --quiet
fi

# ── 1. CUDA arch for RTX 4090 (Ada Lovelace, sm_89) ──
export TORCH_CUDA_ARCH_LIST="8.9"
if ! grep -q "TORCH_CUDA_ARCH_LIST" /etc/bash.bashrc 2>/dev/null; then
    echo 'export TORCH_CUDA_ARCH_LIST="8.9"' >> /etc/bash.bashrc
    echo "  TORCH_CUDA_ARCH_LIST=8.9 persisted to /etc/bash.bashrc"
fi

# ── 2. Clone repo ──
REPO_DIR="/workspace/flux-lm"
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/5] Cloning repo..."
    git clone https://github.com/DavidCaastro/flux-lm.git "$REPO_DIR"
else
    echo "[1/5] Repo already exists, pulling latest..."
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR/torch_port"

# ── 3. Install dependencies ──
echo "[2/5] Checking Python dependencies..."
pip3 install --no-cache-dir -r requirements.txt 2>/dev/null || true

# ── 4. Verify GPU ──
echo "[3/5] GPU check..."
python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPU:      {p.name}')
print(f'  VRAM:     {p.total_memory / 1024**3:.1f} GB')
print(f'  Compute:  {p.major}.{p.minor}')
print(f'  SMs:      {p.multi_processor_count}')
print(f'  bf16:     {torch.cuda.is_bf16_supported()}')
"

# ── 5. Corpus ──
CORPUS="/workspace/corpus_100mb.txt"
if [ ! -f "$CORPUS" ]; then
    echo "[4/5] Building 100MB multilingual corpus..."
    python3 "$REPO_DIR/bench/build_corpus.py" --target-mb 100 --out "$CORPUS"
else
    echo "[4/5] Corpus already exists: $(du -h $CORPUS | cut -f1)"
fi

# ── 6. WandB login (optional) ──
echo "[5/5] WandB setup..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "  WandB configured."
else
    echo "  Skipped (set WANDB_API_KEY env var to enable)."
fi

# ── 7. Clean stale kernel cache (recompile with correct arch) ──
KCACHE="$REPO_DIR/torch_port/.kernel_cache"
if [ -d "$KCACHE" ]; then
    echo "  Clearing old kernel cache to recompile for sm_89..."
    rm -rf "$KCACHE"
fi

echo ""
echo "========================================"
echo "  Setup complete! Training commands:"
echo "========================================"
echo ""
echo "  cd $REPO_DIR/torch_port"
echo ""
echo "  # Quick test (5 min):"
echo "  python3 train.py --corpus $CORPUS \\"
echo "      --d 256 --layers 6 --epochs 5 \\"
echo "      --dtype bf16 --parallel"
echo ""
echo "  # Full training (in tmux):"
echo "  tmux new -s train"
echo "  python3 train.py --corpus $CORPUS \\"
echo "      --d 512 --layers 12 --epochs 250 \\"
echo "      --batch-size 64 --dtype bf16 \\"
echo "      --parallel --grad-checkpoint \\"
echo "      --wandb --ckpt-every 25 --print-every 5"
echo ""
echo "  # Detach tmux: Ctrl+B, D"
echo "  # Reattach:    tmux attach -t train"
echo "========================================"
