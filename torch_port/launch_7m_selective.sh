#!/bin/bash
# ============================================================
# Flux LM 7M — Selective Scan fine-tuning (resume from epoch 67)
#
# Objetivo: activar content-dependent decay sobre modelo 7M convergido
# Config: d=1024, L=12, B=64, T=256, lr=1e-4, cosine, bf16
# LR reducido (1e-4 vs 3e-4 original) para no destruir pesos
# convergidos — leccion aprendida del fallo SGDR en 3.5M.
#
# Selective scan: delta_fast_mod y delta_slow_mod estan en el
# checkpoint como zeros. Al resumir, el modelo aprende decays
# variables gradualmente sin perder lo aprendido.
# ============================================================

set -euo pipefail

CORPUS="/workspace/corpus_python.txt"
CKPT="/workspace/flux-lm/torch_port/checkpoints_7m_v3/flux_epoch_0067.pt"
CKPT_DIR="/workspace/flux-lm/torch_port/checkpoints_7m_selective"
LOG="/workspace/train_7m_selective.log"

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
echo " Flux LM 7M — Selective Scan fine-tuning"
echo " Resume: epoch 67 -> 87 (20 epochs)"
echo " LR: 1e-4 (reducido, cosine decay)"
echo " Checkpoint: $CKPT"
echo " Guardado: cada 2 epochs en $CKPT_DIR"
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
    --epochs 87 \
    --schedule cosine \
    --dtype bf16 \
    --parallel \
    --n-corrections 1 \
    --grad-checkpoint \
    --ckpt "$CKPT" \
    --resume \
    --ckpt-every 2 \
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
