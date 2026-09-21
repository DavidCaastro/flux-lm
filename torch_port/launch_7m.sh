#!/bin/bash
# ============================================================
# Flux LM 7M — Training from scratch
# Config: d=1024, L=12, B=64, T=256, lr=3e-4, cosine, bf16
# Corpus: corpus_python.txt (60MB, same as 3.5M runs)
# Target: ~67 epochs in ~36h on RTX 4090
# ============================================================

set -euo pipefail

CORPUS="/workspace/corpus_python.txt"
LOG="/workspace/train_7m.log"
CKPT_DIR="/workspace/flux-lm/torch_port/checkpoints_7m"

# Verify corpus exists
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: Corpus not found: $CORPUS"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

echo "============================================================"
echo " Flux LM 7M — Training from scratch"
echo " d=1024, L=12, params~7M"
echo " Corpus: $CORPUS ($(du -h "$CORPUS" | cut -f1))"
echo " Log: $LOG"
echo " Checkpoints: $CKPT_DIR"
echo "============================================================"

cd /workspace/flux-lm/torch_port

nohup python train.py \
    --d 1024 \
    --layers 12 \
    --corpus "$CORPUS" \
    --seq-len 256 \
    --batch-size 64 \
    --lr 3e-4 \
    --epochs 67 \
    --schedule cosine \
    --dtype bf16 \
    --parallel \
    --n-corrections 1 \
    --grad-checkpoint \
    --ckpt-every 5 \
    --ckpt-dir "$CKPT_DIR" \
    --print-every 1 \
    --num-workers 2 \
    > "$LOG" 2>&1 &

PID=$!
echo "Training PID: $PID"
sleep 10
echo ""
echo "=== First lines of log ==="
tail -30 "$LOG"
echo ""
echo "Process running: $(ps -p $PID -o pid= 2>/dev/null && echo YES || echo NO)"
