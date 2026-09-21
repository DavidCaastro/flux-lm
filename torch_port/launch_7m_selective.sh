#!/bin/bash
# ============================================================
# Flux LM 7M — Corpus scaling test (resume from epoch 67)
#
# Objetivo: evaluar si corpus 200MB desbloquea capacidad del 7M
# Config: d=1024, L=12, B=64, T=256, lr=1e-4, cosine, bf16
# Corpus: 200MB Python (3.3x mas datos que run v3)
# Resume desde epoch 67 (best 7M, test_bpb=1.092)
# ============================================================

set -euo pipefail

CORPUS="/workspace/corpus_python_200mb.txt"
CKPT="/workspace/flux-lm/torch_port/checkpoints_7m_v3/flux_epoch_0067.pt"
CKPT_DIR="/workspace/flux-lm/torch_port/checkpoints_7m_200mb"
LOG="/workspace/train_7m_200mb.log"

# Verificaciones
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: Corpus no encontrado: $CORPUS"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint no encontrado: $CKPT"
    exit 1
fi

mkdir -p "$CKPT_DIR"

echo "============================================================"
echo " Flux LM 7M — Corpus scaling test (200MB)"
echo " Resume: epoch 67 -> 72 (5 epochs)"
echo " LR: 1e-4 (cosine decay)"
echo " Checkpoint: $CKPT"
echo " Guardado: cada epoch en $CKPT_DIR"
echo " Corpus: $CORPUS ($(du -h "$CORPUS" | cut -f1))"
echo " Log: $LOG"
echo "============================================================"

cd /workspace/flux-lm/torch_port

nohup python train.py \
    --d 1024 \
    --layers 12 \
    --corpus "$CORPUS" \
    --seq-len 256 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --max-grad-norm 5.0 \
    --epochs 72 \
    --schedule cosine \
    --dtype bf16 \
    --parallel \
    --n-corrections 1 \
    --grad-checkpoint \
    --ckpt "$CKPT" \
    --resume \
    --ckpt-every 1 \
    --ckpt-dir "$CKPT_DIR" \
    --print-every 1 \
    --num-workers 2 \
    > "$LOG" 2>&1 &

PID=$!
echo "Training PID: $PID"
sleep 10
echo ""
echo "=== Primeras lineas del log ==="
tail -30 "$LOG"
echo ""
echo "Process running: $(ps -p $PID -o pid= 2>/dev/null && echo YES || echo NO)"
