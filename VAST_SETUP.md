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
```
Host vast
    HostName 137.175.76.24
    Port 41982
    User root
    IdentityFile ~/.ssh/id_vast
```

## Configuracion optima para RTX 4090

```bash
# Asegurar CUDA arch para sm_89
export TORCH_CUDA_ARCH_LIST="8.9"

# Config probada:
cd /workspace/flux-lm/torch_port
python3 train.py --corpus /workspace/corpus_100mb.txt \
    --d 512 --layers 12 --epochs 250 \
    --batch-size 64 --seq-len 256 \
    --dtype bf16 --parallel \
    --grad-checkpoint \
    --ckpt-every 25 --print-every 5

# VRAM esperado: ~8-15 GB de 24 GB
# Throughput esperado: ~25-40K tok/s
```

## Checklist pre-ejecucion

1. [ ] Verificar GPU: `nvidia-smi`
2. [ ] Verificar arch: `echo $TORCH_CUDA_ARCH_LIST` (debe ser "8.9")
3. [ ] Matar procesos previos: `pkill -f python` (excepto jupyter)
4. [ ] Limpiar kernel cache viejo: `rm -rf /workspace/flux-lm/torch_port/.kernel_cache`
5. [ ] Verificar corpus: `ls -lh /workspace/corpus_100mb.txt`
6. [ ] Usar tmux: `tmux new -s train`
7. [ ] Monitorear: `watch nvidia-smi` o wandb
