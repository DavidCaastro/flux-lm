# Flux v3 — Kernels CUDA

## Visión general

Tres kernels CUDA JIT-compilados aceleran las operaciones críticas del modelo. Todos usan `torch.utils.cpp_extension.load_inline` con cache en disco para compilación única.

## 1. WHT Butterfly Kernel (`kernels.py`)

Walsh-Hadamard Transform usando butterfly factorization.

| Propiedad | Valor |
|-----------|-------|
| Archivo | `flux/kernels.py` |
| Cache | `.kernel_cache/` |
| Compilación | ~61s (primera vez) |
| Operación | WHT en O(d log d) |
| Layout | (B*T, d) → (B*T, d) |
| Threads | d per block |
| Límite | d ≤ 1024 |

**Arithmética**: float32 interno, load/store en dtype original (bf16/fp16/fp32).

## 2. Constant-Decay Parallel Scan (`kernels.py`)

Scan recurrente con decay constante (usado por SPM).

```
y[t] = λ * y[t-1] + x[t]    donde λ es constante por dimensión
```

| Propiedad | Valor |
|-----------|-------|
| Archivo | `flux/kernels.py` |
| Cache | `.kernel_cache/` |
| Algoritmo | Hillis-Steele associative scan |
| Layout | (B*D, T) pre-transpuesto |
| Threads | T per block |
| Shared mem | 2 × T × sizeof(float) |
| Límite | T ≤ 1024 |
| Speedup | ~5x vs PyTorch |

**Autograd**: backward computa grad_x via scan reverso + grad_decay analítico (nunca None).

## 3. Variable-Decay Selective Scan (`selective_scan_kernel.py`)

Scan recurrente con decay variable por timestep (selective scan).

```
y[t] = λ(t) * y[t-1] + x[t]    donde λ(t) varía por (batch, time, dim)
```

| Propiedad | Valor |
|-----------|-------|
| Archivo | `flux/selective_scan_kernel.py` |
| Cache | `.selective_scan_cache/` |
| Compilación | ~30s (primera vez) |
| Algoritmo | Hillis-Steele con operador asociativo general |
| Operador | (a₁,b₁) ⊕ (a₂,b₂) = (a₂·a₁, a₂·b₁ + b₂) |
| Layout | (B*D, T) pre-transpuesto |
| Threads | T per block |
| Shared mem | 2 × T × sizeof(float) |
| Límite | T ≤ 1024 |
| Speedup | **7.27x** vs PyTorch |
| Benchmark | 0.45ms kernel vs 3.24ms PyTorch (B=64, T=256, D=512) |

**Autograd backward**:
- `grad_x`: scan reverso usando decay[t+1] (shifted), reutiliza el mismo kernel forward
- `grad_decay[t] = grad_x[t] * y[t-1]`: computado en float32 para evitar overflow bf16
- **Nunca retorna None** para ningún gradiente (lección del bug 995e7f1)

**Fallback**: Si T > 1024, usa automáticamente la implementación PyTorch pura.

## Compilación

Todos los kernels se compilan para **sm_89** (RTX 4090 Ada Lovelace):

```
-gencode arch=compute_89,code=sm_89
-gencode arch=compute_89,code=compute_89  (PTX para forward compat)
```

Flags adicionales: `-O3`, `--use_fast_math`

### Recompilación

Limpiar caches cuando se modifica el código CUDA:

```bash
rm -rf .kernel_cache/ .selective_scan_cache/ flux/__pycache__/
```

## Tests

```bash
# En GPU (vast.ai):
python test_selective_scan_kernel.py
```

7 tests:
1. Forward correctness vs referencia secuencial (tol 1e-3)
2. bf16 forward sin NaN
3. Backward grad nunca None
4. Backward correctness vs PyTorch autograd
5. NaN safety con decay extremo (0.001, 0.999)
6. gradcheck float32
7. Benchmark speedup

Resultado: **7/7 PASS**, speedup 7.27x.

## Guardias anti-NaN

| Riesgo | Guardia |
|--------|---------|
| Producto acumulado underflow | Aritmética float32 interna (min subnormal ~1.4e-45) |
| grad_decay overflow en bf16 | Computado en float32, cast a dtype al final |
| grad_decay = None | Test explícito, backward siempre retorna 2 gradientes |
| T > 1024 threads | TORCH_CHECK + fallback automático a PyTorch |
| Kernel launch error | CUDA_CHECK_LAST_ERROR después de cada launch |
