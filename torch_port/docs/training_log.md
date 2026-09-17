# Flux v3 — Registro de Entrenamiento

## Configuración base

| Parámetro | Valor |
|-----------|-------|
| Modelo | d=512, L=12, ~3.51M params |
| Corpus | corpus_python.txt (60 MB, 26 repos GitHub) |
| Batch size | 64 |
| Seq length | 256 |
| Precisión | bf16 (AMP) |
| Optimizer | EntropicAdam (lr=3e-4, weight_decay=1e-5) |
| LR schedule | WarmRestartCosineSchedule (T₀=20 epochs, T_mult=2) |
| GPU | NVIDIA RTX 4090 (24 GB VRAM, sm_89) |
| Plataforma | Vast.ai, NGC PyTorch nv26.08 container |

## Métricas

- **test_loss**: cross-entropy en nats (base e)
- **BPB** (bits per byte): test_loss / ln(2) ≈ test_loss / 0.6931
- **lr_scale**: factor del schedule coseno [0, 1]

---

## Run 1 — Baseline con decay constante
**Epochs 1–30 | 2026-09-15/16 | ~16 min/epoch**

Parallel scan con decay fijo (λ constante por dimensión).

| Epoch | train_bpb | test_bpb | Notas |
|-------|-----------|----------|-------|
| 1 | 3.05 | 2.07 | Inicio |
| 10 | — | ~1.30 | Descenso rápido |
| 20 | — | ~1.10 | Desacelerando |
| 25 | — | ~1.05 | Estancamiento |
| 30 | 1.185 | 1.046 | Fin Run 1 |

**Resultado**: Estancamiento en BPB ~1.046 desde epoch 25. Mejora residual Δ≈0.003/epoch.

**Diagnóstico**: Decay fijo impone un compromiso irreconciliable — las dimensiones "fast" no pueden capturar dependencias largas, las "slow" no reaccionan a cambios locales. El modelo necesita decay adaptativo.

---

## Run 2 — Selective scan (PyTorch puro)
**Epochs 31–45 | 2026-09-16 | ~47 min/epoch**

Implementación del selective scan: decay depende del contenido (`δ + h·δ_mod`).
Sin kernel CUDA — scan en PyTorch puro (log₂(T) iteraciones con F.pad).

- Velocidad: 0.81s/batch, ~20k tok/s
- Recuperación: 0.016 BPB/epoch (5x más rápido que el baseline)
- Bottleneck: 85% del tiempo en parallel scan PyTorch

**Resultado**: Rompió estancamiento. BPB 1.046 → mejorando ~0.016/epoch.

---

## Run 3 — Selective scan + kernel CUDA
**Epochs 46–50 | 2026-09-17 | ~17 min/epoch**

Kernel CUDA fusionado para variable-decay scan (7.27x speedup sobre PyTorch).

| Epoch | test_loss | BPB | lr_scale |
|-------|-----------|-----|----------|
| 46 | 0.669 | 0.965 | 0.016 |
| 47 | 0.668 | 0.963 | 0.009 |
| 48 | 0.668 | 0.963 | 0.004 |
| 49 | 0.668 | 0.963 | 0.001 |
| 50 | 0.668 | 0.963 | 0.000 |

**Resultado**: Convergencia al fondo del ciclo 1 SGDR. BPB = 0.963.

**Mejora total vs baseline**: 1.046 → 0.963 = **-7.9%** gracias al selective scan.

---

## Run 4 — SGDR warm restart (FALLIDO)
**Epoch 51+ | 2026-09-17 | Matado tras 3 epochs**

Warm restart del LR schedule. test_loss degradó 0.668 → 0.840 en 3 epochs.

### Bugs encontrados y corregidos

**Bug 1 — Weight decay sin escalar por lr_scale:**
```python
# ANTES (incorrecto):
weight_decay = args.weight_decay  # constante 1e-5
# Cuando lr→0, los pesos se decaen sin compensación del optimizer

# DESPUÉS (correcto):
wd_eff = args.weight_decay * lr_scale
```

**Bug 2 — sign_history (int64) corrupto por load_state_dict:**
```python
# EntropicAdam usa sign_history como int64 con operaciones bitwise
# load_state_dict() convierte int64 → float32, perdiendo precisión >2^24

# Fix: backup int64 antes de load, restore después
sign_backup = {k: v.clone() for k, v in opt_state if v.dtype == torch.int64}
optimizer.load_state_dict(ckpt['optimizer'])
# restore int64 values
for k, v in sign_backup.items():
    optimizer.state[k]['sign_history'] = v
```

---

## Run 5 — SGDR con fixes aplicados (MATADO)
**Epochs 46–77 | 2026-09-17 | ~17 min/epoch**

Reanudado desde epoch 45 con ambos fixes. Resultado completo:

### Ciclo 1 SGDR (epochs 46–50) — OK

| Epoch | test_loss | BPB | lr_scale |
|-------|-----------|-----|----------|
| 46 | 0.669 | 0.965 | 0.016 |
| 47 | 0.668 | 0.963 | 0.009 |
| 48 | 0.668 | 0.963 | 0.004 |
| 49 | 0.668 | 0.963 | 0.001 |
| 50 | **0.668** | **0.963** | 0.000 |

### Ciclo 2 SGDR (epochs 51–70) — FALLÓ

| Epoch | test_loss | BPB | lr_scale | Fase |
|-------|-----------|-----|----------|------|
| 51 | 0.678 | 0.978 | 0.999 | Warm restart |
| 52 | 0.679 | 0.980 | 0.999 | Exploración |
| 53 | 0.682 | 0.984 | 0.998 | Exploración |
| 54 | 0.685 | 0.988 | 0.996 | Exploración |
| 55 | 0.687 | 0.991 | 0.994 | Exploración |
| 56 | 0.695 | 1.002 | 0.991 | Exploración |
| 57 | 0.690 | 0.996 | 0.988 | Micro-descenso |
| 58 | 0.696 | 1.004 | 0.984 | Rebote |
| 59 | 0.702 | 1.013 | 0.980 | Subiendo |
| 60 | 0.701 | 1.011 | 0.976 | — |
| 61 | 0.705 | 1.017 | 0.971 | — |
| 62 | 0.714 | 1.030 | 0.965 | — |
| 63 | 0.713 | 1.028 | 0.959 | — |
| 64 | 0.713 | 1.028 | 0.952 | Mínimo C2 |
| 65 | 0.712 | 1.027 | 0.946 | Plateau |
| 66 | 0.717 | 1.034 | 0.938 | — |
| 67 | 0.721 | 1.040 | 0.930 | — |
| 68 | 0.725 | 1.046 | 0.922 | — |
| 69 | 0.724 | 1.045 | 0.914 | — |
| 70 | 0.733 | 1.058 | 0.905 | Fondo C2 |

**El ciclo 2 nunca recuperó el mínimo de 0.668.** Mínimo del ciclo: 0.712 (epoch 64-65).

### Ciclo 3 SGDR (epochs 71–77) — Empeoró

| Epoch | test_loss | BPB | lr_scale |
|-------|-----------|-----|----------|
| 71 | 0.730 | 1.053 | 0.885 |
| 72 | 0.729 | 1.052 | 0.875 |
| 73 | 0.736 | 1.062 | 0.864 |
| 74 | 0.737 | 1.063 | 0.854 |
| 75 | 0.734 | 1.059 | 0.842 |
| 76 | 0.742 | 1.071 | 0.831 |
| 77 | — | — | ~0.820 |

**Matado en epoch 77.** grad_norm=43 (vs ~6-10 normal), modelo inestable.

### Análisis del fallo SGDR

**Causa raíz**: LR pico de 3e-4 es demasiado alto para un modelo de 3.5M params que ya convergió. El warm restart destruye la estructura de pesos aprendida más rápido de lo que el modelo puede reconstruirla durante el descenso del coseno.

**Evidencia**:
- Ciclo 2 (20 epochs) nunca bajó del mínimo del ciclo 1
- Ciclo 3 (40 epochs) empezó incluso peor que donde terminó el ciclo 2
- Degradación acumulativa: cada restart destruye más y recupera menos
- grad_norm se dispara (43 vs 6-10), señal de landscape inestable

**Lección**: SGDR requiere LR pico decreciente por ciclo (ej. 3e-4 → 1e-4 → 3e-5), o el modelo debe tener capacidad suficiente para absorber la perturbación.

---

## Resumen de resultados

| Run | Epochs | Mejor BPB | Mejora | Velocidad |
|-----|--------|-----------|--------|-----------|
| 1 — Baseline | 1-30 | 1.046 | — | 16 min/ep |
| 2 — Selective scan | 31-45 | ~0.98 | +6.3% | 47 min/ep |
| 3 — + Kernel CUDA | 46-50 | **0.963** | +7.9% | 17 min/ep |
| 4 — SGDR (buggy) | 51-53 | — | FALLIDO | — |
| 5 — SGDR (fixed) | 46-77 | 0.963 | +0% | 17 min/ep |

**Mejor resultado global: BPB = 0.963** (test_loss = 0.668, epoch 50)

### Checkpoints disponibles en vast.ai
```
/workspace/flux-lm/torch_port/checkpoints/
├── flux_epoch_0030.pt  (~42 MB)
├── flux_epoch_0035.pt
├── flux_epoch_0040.pt
├── flux_epoch_0045.pt
├── flux_epoch_0050.pt  ← MEJOR
├── flux_epoch_0055.pt
├── flux_epoch_0060.pt
├── flux_epoch_0065.pt
├── flux_epoch_0070.pt
└── flux_epoch_0075.pt
```

---

## Próximos pasos posibles

1. **SWA (Stochastic Weight Averaging)**: Promediar checkpoints 45/50 para ~0.02 BPB gratis
2. **Más datos**: Reentrenar desde epoch 50 con corpus_1gb_en_es.txt + LR más bajo (1e-4)
3. **Escalar modelo**: d=512→768/1024 (~14M params), mejora esperada ~0.15-0.25 BPB
4. **Shampoo optimizer**: Segundo orden, mejores mínimos pero 30-50% más costoso

### Límites teóricos
- Shannon floor para Python code (modelo infinito): ~0.3-0.5 BPB
- Estimación práctica para 3.5M params: ~0.6-0.7 BPB
- Gap actual: 0.963 - 0.7 ≈ 0.26 BPB de mejora posible
