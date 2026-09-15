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

## Configuracion optima para RTX 4090

```bash
# Config probada y verificada:
python train.py --corpus /workspace/corpus_100mb.txt \
    --d 512 --layers 12 --epochs 250 \
    --batch-size 64 --seq-len 256 \
    --dtype bf16 --parallel \
    --grad-checkpoint \
    --ckpt-every 25 --print-every 5

# VRAM esperado: ~8-15 GB de 24 GB
# Throughput esperado: ~25-40K tok/s
# Tiempo por epoch: ~50-70 min
```

## Checklist pre-ejecucion

1. [ ] Verificar GPU: `nvidia-smi`
2. [ ] Matar procesos previos: `pkill -f python` (excepto jupyter)
3. [ ] Verificar corpus: `ls -lh /workspace/corpus_100mb.txt`
4. [ ] Usar tmux: `tmux new -s train`
5. [ ] Usar nohup si no hay tmux: `nohup python train.py ... > train.log 2>&1 &`
6. [ ] Monitorear: `tail -f train.log` o wandb
