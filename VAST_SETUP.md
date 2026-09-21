# Flux LM — Vast.ai Setup & Troubleshooting

## Problemas encontrados y soluciones aplicadas

### 1. EntropicAdam: loop CPU bloqueante (CRITICO)

**Problema:** El optimizer original tenia un `for` loop Python con 54,687 iteraciones
para d=512. Cada iteracion llamaba `.item()` que fuerza sincronizacion GPU→CPU.
Resultado: ~270ms por optimizer step, casi igual que forward+backward combinados.

**Solucion:** Vectorizacion completa en GPU:
- Sign history almacenado como tensor int64 en GPU
- Popcount via bit-parallel (Hamming weight sin loops)
- Entropia y LR per-group calculados con operaciones tensoriales
- Cero llamadas `.item()`, cero sincronizaciones CPU

Archivo: `torch_port/flux/optim.py`

### 2. Train.py: sin cleanup ni signal handling

**Problema:** Si el proceso recibia SIGTERM (instancia spot preempted, SSH cortado),
no guardaba checkpoint y dejaba procesos huerfanos con GPU memoria ocupada.

**Solucion:**
- Signal handler para SIGTERM/SIGINT que guarda checkpoint antes de salir
- `torch.cuda.empty_cache()` al finalizar
- Preflight check automatico antes de entrenar (forward+backward de prueba)
- `persistent_workers` en test DataLoader

Archivo: `torch_port/train.py`

### 3. Procesos huerfanos en GPU

**Problema:** Scripts de diagnostico que crasheaban dejaban procesos Python
con VRAM reservada. Ejecuciones posteriores fallaban con OOM.

**Solucion:** Antes de lanzar entrenamiento, verificar y limpiar:
```bash
# Ver procesos en GPU
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv

# Matar procesos huerfanos
pkill -f 'python.*train'
```

El nuevo train.py limpia automaticamente al salir.

### 4. Subutilizacion GPU (0.09% eficiencia compute)

**Causa raiz:** WHT y parallel_scan usan Python while-loops que lanzan ~2,206
kernels CUDA pequenos por forward pass. Cada kernel launch tiene ~10us overhead.
Tensores de 8.4MB cuando GPU necesita >100MB para saturar bandwidth.

**Estado:** Esto es una limitacion de la arquitectura recurrente en PyTorch puro.
Soluciones posibles para el futuro:
- Triton kernels fusionados (eliminaria ~2000 kernel launches)
- torch.compile con mode="reduce-overhead"
- Aumentar batch_size con gradient checkpointing

### 5. torch.compile: overhead excesivo

**Problema:** Primera epoch tarda ~8 min compilando kernels CUDA.
Para modelos pequenos (<10M params) el overhead no compensa.

**Recomendacion:** NO usar --compile para runs cortos (<100 epochs).
Solo vale la pena para runs muy largos (>500 epochs) donde el costo
de compilacion se amortiza.

## Instancia Vast.ai — Hardware verificado (2026-09-15)

| Componente | Detalle |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89, 23.5 GB, 128 SMs) |
| Driver | 580.159.03 |
| CUDA (driver) | 13.4 |
| CUDA Toolkit (nvcc) | 13.4 V13.4.59 |
| PyTorch | 2.14.0a0+nv26.08 (NGC custom) |
| cuDNN | 92500 |
| Triton | 3.8.0+nv26.8 |
| Python | 3.12.3 |
| OS | Ubuntu 24.04.4 LTS |
| CPU | AMD EPYC 7542 (128 threads) |
| RAM | 251 GB |
| Disco | 32 GB overlay |
| gcc/g++ | 13.3.0 |
| ninja | 1.13.0 |
| wandb | 0.28.2 |
| tmux | 3.4 |

### SSH

> **NOTA**: IP y puerto cambian con cada instancia. Actualizar tras crear/reiniciar.
> Usar `vastai show instances` para obtener datos actuales.

```
Host vast
    HostName <IP_INSTANCIA>
    Port <PUERTO>
    User root
    IdentityFile ~/.ssh/id_vast
    StrictHostKeyChecking no
    ServerAliveInterval 30
    ServerAliveCountMax 10
```

## Configuraciones de entrenamiento verificadas

```bash
# Asegurar CUDA arch para sm_89
export TORCH_CUDA_ARCH_LIST="8.9"
cd /workspace/flux-lm/torch_port
```

### Modelo 3.5M (d=512) — BPB alcanzado: 0.963

```bash
python3 train.py --corpus /workspace/corpus_python.txt \
    --d 512 --layers 12 --epochs 67 \
    --batch-size 64 --seq-len 256 --lr 3e-4 \
    --weight-decay 1e-5 --max-grad-norm 5.0 \
    --dtype bf16 --parallel --n-corrections 1 \
    --grad-checkpoint --schedule cosine \
    --ckpt-every 5 --ckpt-dir checkpoints \
    --print-every 1 --num-workers 2

# VRAM: ~6.8 GB | Throughput: ~25K tok/s | ~47 min/epoch (60MB corpus)
```

### Modelo 7M (d=1024) — BPB alcanzado: 1.092

```bash
python3 train.py --corpus /workspace/corpus_python_200mb.txt \
    --d 1024 --layers 12 --epochs 72 \
    --batch-size 64 --seq-len 256 --lr 1e-4 \
    --weight-decay 1e-5 --max-grad-norm 5.0 \
    --dtype bf16 --parallel --n-corrections 1 \
    --grad-checkpoint --schedule cosine \
    --ckpt-every 1 --ckpt-dir checkpoints_7m \
    --print-every 1 --num-workers 2

# VRAM: ~10 GB | Throughput: ~20K tok/s | ~158 min/epoch (200MB corpus)
```

### Resume desde checkpoint

```bash
# Agregar --ckpt y --resume a cualquier config anterior
python3 train.py [mismos args] \
    --ckpt /workspace/.../flux_epoch_NNNN.pt --resume
```

### Multi-GPU (DDP)

```bash
# N GPUs — reemplazar python3 por torchrun
torchrun --nproc_per_node=4 train.py [mismos args]

# NOTA: con 4 GPUs el batch efectivo es 4×64=256
# Si batch > 128, considerar subir LR proporcionalmente (linear scaling rule)
```

### Corpus disponibles (verificados 2026-09-21)

| Archivo | Tamanio | Batches/epoch (B=64) | Tiempo/epoch (1×4090) |
|---------|---------|---------------------|----------------------|
| corpus_python.txt | 60 MB | 3,457 | ~47 min |
| corpus_python_200mb.txt | 200 MB | 11,520 | ~158 min |
| corpus_python_536mb.txt | 536 MB | 30,858 | ~422 min |

## Compatibilidad de checkpoints — IMPORTANTE

El modelo ha evolucionado entre runs. Checkpoints viejos pueden ser incompatibles con codigo nuevo:

| Checkpoint | Codigo compatible | Parametros exclusivos |
|------------|-------------------|-----------------------|
| 3.5M epoch 50 | `model_old.py` (pre-commit `53bafd6`) | NO tiene: spm_w_dec, spm_delta_mod, w_c_fast, w_c_slow |
| 7M epoch 67+ | `model.py` actual | Tiene todos los params actuales |

**Si cargas un checkpoint viejo con codigo nuevo**, `strict=False` inicializa params faltantes — pero `spm_w_dec` se inicializa aleatorio y el output multiplicativo (`(1+cond)*base`) destruye las predicciones. Ver `docs/training_analysis.md` seccion 11.2 para detalles.

**Regla**: si haces cambios arquitectonicos en `model.py`, verifica que checkpoints existentes siguen funcionando antes de commitear.

## Checklist pre-ejecucion

1. [ ] Verificar GPU: `nvidia-smi`
2. [ ] Verificar arch: `echo $TORCH_CUDA_ARCH_LIST` (debe ser "8.9")
3. [ ] Matar procesos previos: `pkill -f python` (excepto jupyter)
4. [ ] Limpiar kernel cache: `rm -rf /workspace/flux-lm/torch_port/.kernel_cache`
5. [ ] Verificar corpus: `ls -lh /workspace/corpus_python*.txt`
6. [ ] Verificar disco libre: `df -h /workspace` (necesitas N_epochs × ~82MB para checkpoints 7M)
7. [ ] Si resume: verificar compatibilidad checkpoint/codigo (ver tabla arriba)
8. [ ] Codigo actualizado: `cd /workspace/flux-lm && git pull`
9. [ ] Usar tmux: `tmux new -s train`
10. [ ] Monitorear: `tail -f /workspace/train*.log` o wandb

## Referencia rapida de flags de train.py

| Flag | Descripcion | Default | Valores probados |
|------|-------------|---------|-----------------|
| `--d` | Dimension del modelo | 256 | 512 (3.5M), 1024 (7M) |
| `--layers` | Numero de capas | 3 | 12 |
| `--corpus` | Path al corpus | (requerido) | /workspace/corpus_python*.txt |
| `--seq-len` | Longitud de secuencia | 256 | 256 |
| `--batch-size` | Batch size por GPU | 64 | 64 |
| `--lr` | Learning rate | 1e-3 | 3e-4 (3.5M), 1e-4 (7M fine-tune) |
| `--weight-decay` | Weight decay | 1e-5 | 1e-5 |
| `--max-grad-norm` | Gradient clipping | 5.0 | 5.0 |
| `--epochs` | Total epochs | 50 | 67-72 |
| `--schedule` | LR schedule | cosine | cosine, sgdr |
| `--dtype` | Precision | bf16 | bf16 |
| `--parallel` | Parallel scan (vs sequential) | off | siempre activar |
| `--n-corrections` | Perturbative corrections | 1 | 1 |
| `--grad-checkpoint` | Gradient checkpointing | off | siempre activar |
| `--ckpt` | Path a checkpoint para cargar | None | /workspace/.../flux_epoch_NNNN.pt |
| `--resume` | Reanudar optimizer + epoch | off | activar con --ckpt |
| `--ckpt-every` | Guardar cada N epochs | 10 | 1-5 |
| `--ckpt-dir` | Directorio de checkpoints | checkpoints | checkpoints_7m_200mb/ |
| `--print-every` | Print cada N epochs | 5 | 1 |
| `--num-workers` | DataLoader workers | 2 | 2 |
| `--wandb` | Activar wandb logging | off | opcional |
| `--compile` | torch.compile | off | NO recomendado (<100 epochs) |
| `--early-stop-bpb` | Early stop target | None | opcional |
