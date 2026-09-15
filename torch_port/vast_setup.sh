#!/bin/bash
# ============================================================
# Flux LM — Vast.ai instance setup script
# Run this ONCE after SSH into your Vast.ai instance
# ============================================================
set -e

echo "========================================"
echo "  Flux LM — Vast.ai Setup"
echo "========================================"

# ── 1. System basics ──
apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip tmux htop nvtop 2>/dev/null || true

# ── 2. Clone repo ──
REPO_DIR="/workspace/flux-lm"
if [ ! -d "$REPO_DIR" ]; then
    echo "[1/5] Cloning repo..."
    git clone https://github.com/YOUR_USER/flux-lm.git "$REPO_DIR"
else
    echo "[1/5] Repo already exists, pulling latest..."
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR/torch_port"

# ── 3. Install dependencies ──
echo "[2/5] Installing Python dependencies..."
pip3 install --no-cache-dir -r requirements.txt

# ── 4. Verify GPU ──
echo "[3/5] GPU check..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPUs:    {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'  [{i}] {name} ({mem:.0f} GB)')
print(f'bf16:    {torch.cuda.is_bf16_supported()}')
print(f'compile: {hasattr(torch, \"compile\")}')
"

# ── 5. Build corpus (if not present) ──
CORPUS="/workspace/flux-lm/bench/corpus_multilingual_100mb.txt"
if [ ! -f "$CORPUS" ]; then
    echo "[4/5] Building 100MB multilingual corpus..."
    python3 ../bench/build_corpus.py --target-mb 100 --out "$CORPUS"
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
    echo "  Get your key at: https://wandb.ai/authorize"
fi

echo ""
echo "========================================"
echo "  Setup complete! Run training with:"
echo "========================================"
echo ""
echo "  # Quick test (5 min):"
echo "  python3 train.py --corpus $CORPUS \\"
echo "      --d 256 --layers 6 --epochs 5 \\"
echo "      --dtype bf16 --compile --parallel"
echo ""
echo "  # Full training:"
echo "  tmux new -s train"
echo "  python3 train.py --corpus $CORPUS \\"
echo "      --d 512 --layers 12 --epochs 300 \\"
echo "      --batch-size 64 --dtype bf16 \\"
echo "      --compile --parallel --wandb \\"
echo "      --ckpt-every 25 --print-every 5"
echo ""
echo "  # Detach tmux: Ctrl+B, D"
echo "  # Reattach:    tmux attach -t train"
echo "========================================"
