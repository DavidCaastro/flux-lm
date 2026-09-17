# Flux v3 — Arquitectura del Modelo

## Visión general

Flux v3 es un **byte-level language model** que opera directamente sobre bytes (0-255) sin tokenización. Combina transformaciones algebraicas eficientes (WHT) con memoria recurrente de doble escala temporal y gating dependiente del contenido.

```
Input bytes → Embedding(256, d) → L × FluxLayer → Output(d, 256) → softmax → next byte
```

## Parámetros del modelo entrenado

| Parámetro | Valor |
|-----------|-------|
| d (dimensión) | 512 |
| L (capas) | 12 |
| K (particiones SPM) | 4 |
| Total params | ~3.51M |
| Vocab | 256 (byte-level) |

## Componentes por capa (FluxLayer)

### 1. Walsh-Hadamard Transform (WHT)
Mezcla de dimensiones sin parámetros aprendibles, O(d log d).

```
WHT(x) = H_d · x    donde H es la matriz de Hadamard normalizada
```

- Implementación: butterfly factorization en CUDA (kernel fusionado)
- Equivalente a una capa de mezcla tipo MLP pero sin parámetros
- Normalización: 1/sqrt(d) para mantener escala

### 2. Content-Dependent Gating
Gate por byte que modula la señal después del WHT.

```
gate = sigmoid(W_gate · x + b_gate)
x = x * gate
```

- W_gate: (d, d), b_gate: (d,)
- Permite al modelo seleccionar qué dimensiones son relevantes para cada byte

### 3. Dual-Timescale Recurrent Memory
Dos filtros exponenciales paralelos con decay rates diferentes:

```
h_fast[t] = λ_fast(t) * h_fast[t-1] + (1 - λ_fast(t)) * x[t]
h_slow[t] = λ_slow(t) * h_slow[t-1] + (1 - λ_slow(t)) * x[t]
```

**Decay learnable:**
```
λ(t) = exp(-softplus(δ + h[t-1] · δ_mod))
```

- `δ` (delta): parámetro base por dimensión (d,)
- `δ_mod` (delta_mod): modulación dependiente del contenido (d,) — **selective scan**
- Init: δ con ramp logarítmica, δ_mod = zeros (comportamiento inicial = decay constante)

**Fast vs Slow:**
- δ_fast: ramp [0, 5] → decays altos, memoria corta
- δ_slow: ramp [-5, 0] → decays bajos, memoria larga

**Implementación:** parallel scan asociativo (Hillis-Steele) con kernel CUDA fusionado.

### 4. Semantic Partition Module (SPM)
Memoria semántica adicional con K=4 particiones de stride 4.

```
h_sem[k] = λ_sem * h_sem[k][t-1] + (1 - λ_sem) * x[k*stride:(k+1)*stride]
```

- Decay constante (no content-dependent)
- Gate SPM init: -5.0 (señal semántica casi apagada al inicio)
- Usa kernel CUDA de scan constante (kernels.py)

### 5. Combinación y residual

```
h_combined = W_mix · concat(h_fast, h_slow, h_sem) + b_mix
output = x + h_combined * (1 / ln(L + 2))
```

- Residual scaling 1/ln(L+2) para estabilidad en redes profundas

## Selective Scan (v2) — Innovación clave

El decay content-dependent es la diferencia principal vs el modelo Rust original:

| Aspecto | Rust (original) | PyTorch (actual) |
|---------|-----------------|------------------|
| Decay | Fijo: `λ = exp(-softplus(δ))` | Variable: `λ(t) = exp(-softplus(δ + h·δ_mod))` |
| Scan | Secuencial (CPU) | Parallel scan (GPU CUDA kernel) |
| Efecto | Compromiso fijo fast/slow | El modelo decide cuánto recordar/olvidar |

Esto rompió el estancamiento en BPB 1.046 → 0.963 (mejora de 7.9%).

## Operador asociativo del parallel scan

El scan usa el operador:
```
(a₁, b₁) ⊕ (a₂, b₂) = (a₂·a₁, a₂·b₁ + b₂)
```

donde `a = decay`, `b = input`. Esto permite paralelizar el scan recurrente en O(T log T) usando el algoritmo de Hillis-Steele.

## Kernels CUDA (JIT compilados)

| Kernel | Archivo | Función | Speedup |
|--------|---------|---------|---------|
| WHT butterfly | kernels.py | Mezcla O(d log d) | ~3x vs PyTorch |
| Constant-decay scan | kernels.py | SPM memory | ~5x vs PyTorch |
| Variable-decay scan | selective_scan_kernel.py | Fast/slow memory | **7.27x** vs PyTorch |

Todos compilados para sm_89 (RTX 4090) con fallback a PyTorch si T > 1024.
