# Analisis Tecnico del Entrenamiento — Flux LM

> Documento de referencia interna. Datos verificados contra logs de entrenamiento.
> Revision: 2026-09-21

---

## 1. Configuraciones Experimentales

### 1.1 Modelo 3.5M (d=512)

| Parametro | Valor |
|-----------|-------|
| Dimension d | 512 |
| Capas L | 12 |
| Parametros totales | ~3,500,000 |
| Vocabulario | 256 (byte-level) |
| Embedding | 256 x 512 = 131,072 |
| Head (output) | 256 x 512 + 256 = 131,328 |
| SPM projection | K x d = 4 x 512 = 2,048 |
| Params por capa | ~270,000 |

**Desglose por capa (FluxLayer, d=512):**

| Componente | Parametros | Formula |
|------------|-----------|---------|
| rn_gamma | 512 | d |
| g_gate (Embedding 256xd) | 131,072 | 256 * d |
| a_bias (Embedding 256xd) | 131,072 | 256 * d |
| w_gate_h | 512 | d |
| s1, b1, s2, b2 | 2,048 | 4 * d |
| delta_fast, delta_fast_mod | 1,024 | 2 * d |
| delta_slow, delta_slow_mod | 1,024 | 2 * d |
| b_in_fast, b_in_slow | 1,024 | 2 * d |
| c_out_fast, c_out_slow | 1,024 | 2 * d |
| w_c_fast, w_c_slow | 1,024 | 2 * d |
| skip | 512 | d |
| spm_delta | 4 | K |
| spm_gate | 512 | d |
| **Total por capa** | **~270,364** | |

### 1.2 Modelo 7M (d=1024)

| Parametro | Valor |
|-----------|-------|
| Dimension d | 1024 |
| Capas L | 12 |
| Parametros totales | 7,041,328 |
| Embedding | 256 x 1024 = 262,144 |
| Head | 256 x 1024 + 256 = 262,400 |
| Params por capa | ~543,000 |

### 1.3 Corpus

- **Fuente**: 26 repositorios open-source de Python (GitHub)
- **Tamanio**: 56.6 MB (corpus_python.txt) para 7M; 60 MB para 3.5M
- **Split**: 90% train / 10% test
- **Train chunks**: 221,248 (7M) / ~220,000 (3.5M)
- **Test chunks**: 24,583 (7M)
- **Batches/epoch**: 3,457 (7M, B=64)
- **Tokens/epoch**: 56,599,552 (3,457 batches x 64 x 256)
- **Secuencia**: T = 256 bytes

### 1.4 Hiperparametros de entrenamiento

| Parametro | 3.5M (Runs 1-5) | 7M (Run v3) |
|-----------|-----------------|-------------|
| Batch size B | 64 | 64 |
| Seq length T | 256 | 256 |
| Learning rate | 3e-4 | 3e-4 |
| Optimizer | EntropicAdam | EntropicAdam |
| beta1, beta2 | 0.9, 0.999 | 0.9, 0.999 |
| Weight decay | 1e-5 | 1e-5 |
| Max grad norm | 5.0 | 5.0 |
| Precision | bf16 (AMP) | bf16 (AMP) |
| Schedule | cosine / SGDR | cosine |
| Epochs | 50 (cosine) / 77 (SGDR) | 67 |
| Warmup | 5% de epochs | 5% de epochs |
| Grad checkpoint | Si | Si |
| n_corrections | 1 | 1 |

---

## 2. Optimizer: EntropicAdam

### 2.1 Formulacion matematica

EntropicAdam extiende Adam con un learning rate modulado por la entropia de la historia de signos del gradiente. Para cada grupo de `gs` parametros:

**Paso 1 — Momentos Adam estandar:**

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

**Paso 2 — Historia de signos:**

Se mantiene un registro de 32 bits (`sign_history`) por grupo. En cada paso:

```
majority_t = 1  si  sum(g_t[grupo] > 0) >= gs/2
             0  en otro caso

sign_history = (sign_history << 1) & 0xFFFFFFFF | majority_t
```

**Paso 3 — Entropia binaria:**

```
bits = popcount(sign_history)
p = bits / 32
H(p) = -(p * log(p) + (1-p) * log(1-p))
```

donde H(p) in [0, ln(2)] con maximo en p = 0.5.

**Paso 4 — Learning rate modulado:**

```
lr_base = lr * sqrt(1 - beta2^t) / (1 - beta1^t)    [correccion de sesgo Adam]
T_epoch = T_initial * (T_final / T_initial)^(epoch / total_epochs)
glr = lr_base * exp(-H(p) / T_epoch)
```

**Paso 5 — Actualizacion:**

```
theta_t = theta_{t-1} - glr * m_t / (sqrt(v_t) + eps)
```

### 2.2 Interpretacion

- Cuando H(p) es alto (signos aleatorios, gradiente ruidoso): `exp(-H/T)` es pequeno, el LR efectivo se reduce. El optimizador es conservador en direcciones inciertas.
- Cuando H(p) es bajo (signos consistentes, gradiente coherente): `exp(-H/T) ~ 1`, el LR efectivo es maximo. El optimizador avanza agresivamente en direcciones claras.
- `T_epoch` controla la sensibilidad: T alto -> todos los LR similares; T bajo -> diferenciacion agresiva entre direcciones ciertas e inciertas.
- La temperatura decae exponencialmente de `T_initial=2.0` a `T_final=0.5` durante el entrenamiento, haciendo al optimizador progresivamente mas selectivo.

### 2.3 Parametros: group_size=64

Con `gs=64`, se agrupan 64 parametros contiguos en memoria. El voto mayoritario reduce ruido estocástico vs. tracking por parametro individual. Para un modelo de 3.5M params, esto produce ~54,688 grupos de LR independientes.

---

## 3. Arquitectura del Modelo

### 3.1 Flujo de datos (forward pass, modo paralelo)

Para una secuencia de bytes `x` de forma `(B, T)`:

```
x_emb = Embedding(x)                             # (B, T, d)

Para cada capa l = 0..L-1:
  1. RMSNorm:     state = x / sqrt(mean(x^2) + eps) * gamma
  2. Gating:      gate = sigma(g_gate(byte_id) + w_gate_h * h_fast_{t-1})
  3. State mix:   state = gate * state + a_bias(byte_id)
  4. WHT block 1: state = tanh(s1 * WHT(state) + b1)
  5. WHT block 2: state = tanh(s2 * WHT(state) + b2)
  6. Decay:       lam_fast_t = exp(-softplus(delta_fast + state * delta_fast_mod))
                  lam_slow_t = exp(-softplus(delta_slow + state * delta_slow_mod))
  7. Scan fast:   h_fast = parallel_scan(lam_fast_t, b_in_fast * state)
  8. Scan slow:   h_slow = parallel_scan(lam_slow_t, b_in_slow * state)
  9. SPM:         h_sem = parallel_scan(lam_sem, (1-lam_sem) * (h_slow[::4] @ W_spm^T))
                  cond = (h_sem @ W_spm) * sigma(spm_gate)
  10. Output:     c_f = sigma(c_out_fast + w_c_fast * state)
                  c_s = sigma(c_out_slow + w_c_slow * state)
                  out = c_f * h_fast + c_s * h_slow + skip * state + res_scale * x + cond

logits = x_final @ W_head^T + b_head                # (B, T, 256)
loss = CrossEntropy(logits, targets)
```

### 3.2 Walsh-Hadamard Transform (WHT)

La WHT opera sobre la ultima dimension (d) via el algoritmo butterfly:

```
Para half = 1, 2, 4, ..., d/2:
    x = reshape(x, ..., d/(2*half), 2, half)
    a, b = x[..., 0, :], x[..., 1, :]
    x = stack([a+b, a-b], dim=-2)
x = x * (1 / sqrt(d))
```

Complejidad: O(d log d) por token. Para d=512: 9 etapas butterfly, 4608 sumas.

La WHT es una transformacion ortogonal (WHT^T = WHT, WHT * WHT = I * d), lo que preserva normas y permite mezcla de features sin parametros adicionales. A diferencia de matrices lineales densas (O(d^2) parametros), la WHT logra mixing global con 0 parametros entrenables (solo `s` y `b` son aprendidos).

### 3.3 Parallel Scan (Blelloch/Hillis-Steele)

**Decay constante** (SPM): y[t] = lambda * y[t-1] + x[t]

Implementado via prefix sum con potencias cuadradas del decay:

```
stride = 1, decay_pow = lambda
while stride < T:
    y += shift(y, stride) * decay_pow
    decay_pow = decay_pow^2
    stride *= 2
```

Complejidad: O(T log T), profundidad O(log T).

**Decay variable** (selective scan, fast/slow tracks):

El operador asociativo general (a1,b1) @ (a2,b2) = (a2*a1, a2*b1+b2):

```
a = decay (B, T, d)
b = input (B, T, d)
stride = 1
while stride < T:
    a_prev = pad(a[:, :-stride], (stride, 0), value=1.0)
    b_prev = pad(b[:, :-stride], (stride, 0))
    b = a * b_prev + b
    a = a * a_prev
    stride *= 2
```

### 3.4 Slow Persistent Memory (SPM)

El SPM opera como un filtro pasa-bajos sobre la representacion interna:

1. Proyeccion: z = h_slow @ W_spm^T, con W_spm de forma (K, d), K=4
2. Subsampling: procesa cada STRIDE=4 timesteps
3. Recurrencia: h_sem[t] = lam_sem * h_sem[t-1] + (1-lam_sem) * z[t]
4. Recondiccion: cond = (h_sem @ W_spm) * sigma(spm_gate)

El decay `lam_sem = exp(-softplus(spm_delta))` es constante (no content-dependent), inicializado alto (spm_delta en [-5, -3]) para capturar tendencias lentas.

### 3.5 Residual scaling

```
res_scale = 1 / log(layer_idx + 2)
```

Esto reduce la contribucion residual en capas profundas:
- Capa 0: 1/log(2) = 1.443
- Capa 5: 1/log(7) = 0.514
- Capa 11: 1/log(13) = 0.390

---

## 4. Resultados Experimentales

### 4.1 Metricas

- **Loss**: Cross-entropy sobre vocabulario de 256 bytes, en nats.
- **BPB** (Bits Per Byte): loss / ln(2). Interpretable como el numero de bits necesarios para codificar un byte.
- **BPB = 8.0**: cota superior trivial (sin compresion).
- **BPB = 1.0**: compresion 8:1 respecto a la representacion raw.

### 4.2 Modelo 3.5M — Trayectoria completa

#### Run 1: Baseline con decay constante (epochs 1-30)

Config: parallel scan con decay fijo (lambda_fast, lambda_slow constantes por dimension).

| Epoch | train_bpb | test_bpb | Delta/epoch |
|-------|-----------|----------|-------------|
| 1 | 3.05 | 2.07 | — |
| 5 | — | ~1.6 | ~0.094 |
| 10 | — | ~1.3 | ~0.060 |
| 20 | — | ~1.1 | ~0.020 |
| 25 | — | ~1.05 | ~0.010 |
| 30 | 1.185 | 1.046 | ~0.003 |

**Observacion**: Estancamiento a partir de epoch ~25 con Delta_test_bpb/epoch ≈ 0.003. La curva de convergencia sigue un decaimiento exponencial tipico:

```
test_bpb(e) ≈ 1.04 + 1.03 * exp(-e/7.5)
```

El ajuste es razonable para e in [1, 30] (R^2 > 0.99), indicando que el plateau de ~1.046 era cercano a la capacidad del modelo con decay fijo.

**Causa raiz del estancamiento**: El decay constante lambda impone un compromiso estatico entre retencion de informacion y velocidad de olvido. Para cualquier valor fijo de lambda, existe un rango optimo de dependencias temporales que puede capturar, pero fuera de ese rango la informacion se disipa (lambda bajo) o contamina (lambda alto).

#### Run 2: Selective scan — PyTorch puro (epochs 31-45)

Reanudado desde epoch 30. Cambio: decay ahora es content-dependent:

```
delta_t = delta_base + state * delta_mod
lambda_t = exp(-softplus(delta_t))
```

Parametros nuevos: `delta_fast_mod` y `delta_slow_mod` por capa (12,288 params, +0.35%). Inicializados a cero para compatibilidad con checkpoint existente.

| Metrica | Run 1 (epoch 25-30) | Run 2 (epoch 31-35) | Factor |
|---------|--------------------|--------------------|--------|
| Delta_test_bpb/epoch | 0.003 | 0.016 | 5.3x |
| Throughput | ~55k tok/s | ~20k tok/s | 0.36x |

La mejora de 5.3x en velocidad de convergencia a costa de 2.75x menos throughput (sin kernel CUDA) resulta en una mejora neta de ~1.9x en convergencia por segundo de wall-clock.

#### Run 3: Selective scan + kernel CUDA (epochs 46-50)

Reanudado desde epoch 45 con kernel CUDA para el scan variable.

| Epoch | test_loss | test_bpb | lr_scale |
|-------|-----------|----------|----------|
| 46 | 0.669 | 0.965 | 0.016 |
| 47 | 0.668 | 0.963 | 0.009 |
| 48 | 0.668 | 0.963 | 0.004 |
| 49 | 0.668 | 0.963 | 0.001 |
| 50 | 0.668 | 0.963 | 0.000 |

Convergencia al minimo del ciclo cosine. Throughput: ~55k tok/s (2.75x vs PyTorch puro).

#### Run 4: SGDR — primer intento (FALLIDO)

Warm restart en epoch 51 con lr_scale = 1.0 (pico). test_loss degradó 0.668 -> 0.840 en 3 epochs.

**Bugs identificados:**
1. Weight decay no escalaba con lr_scale. Cuando LR -> 0 al final del ciclo, el wd fijo domina la actualizacion y destruye pesos.
2. `sign_history` (int64) se corrompia a float32 en `load_state_dict`, causando popcount erroneo y NaN en entropia.

#### Run 5: SGDR con fixes (epochs 46-77)

Fixes aplicados:
- `wd_effective = weight_decay * lr_scale`
- `sign_history` backup/restore como int64 al cargar checkpoints

**Trayectoria por ciclo:**

| Ciclo | Epochs | lr_pico | test_loss min | test_bpb min | vs Ciclo 1 |
|-------|--------|---------|---------------|-------------|------------|
| 1 | 46-50 | 3e-4 (decayendo) | 0.668 | 0.963 | baseline |
| 2 | 51-70 | 3e-4 | 0.712 | 1.027 | +6.6% |
| 3 | 71-77 | 3e-4 (parcial) | 0.734 | 1.059 | +9.9% |

**Analisis del fallo SGDR:**

Sea L_min^(c) la test_loss minima del ciclo c. Si SGDR funciona, esperamos L_min^(c+1) < L_min^(c). Observamos lo contrario:

```
L_min^(1) = 0.668
L_min^(2) = 0.712   (Delta = +0.044, +6.6%)
L_min^(3) = 0.734   (Delta = +0.066 vs ciclo 1, +9.9%)
```

La degradacion es monotona y se acelera: el modelo no solo no mejora, sino que pierde estructura de forma irreversible.

**Diagnostico cuantitativo:**

1. **Tasa de destruccion vs. reconstruccion**: Al aplicar lr_pico = 3e-4 sobre un modelo convergido de 3.5M params, la perturbacion por paso es del orden:

   ```
   ||Delta_theta|| / ||theta|| ≈ lr * ||g|| / ||theta|| ≈ 3e-4 * 10 / 60 ≈ 5e-5
   ```

   En 20 epochs x 3457 pasos = 69,140 pasos con lr > 0.5 * lr_pico, esto acumula una perturbacion total del orden de:

   ```
   sqrt(69140) * 5e-5 ≈ 0.013 (1.3% de ||theta||)
   ```

   Esta perturbacion es suficiente para escapar del minimo local pero insuficiente para encontrar uno mejor en el landscape de un modelo de 3.5M params, que tiene pocos minimos comparativamente profundos.

2. **grad_norm**: En epoch 77, grad_norm = 43, vs ~6-10 en regimen estable. Ratio 4-7x indica inestabilidad del training.

3. **Conclusion**: SGDR asume que el landscape de loss tiene multiples minimos amplios a distancias explorables por el warm restart. Para modelos pequenos (3.5M params), el landscape es mas simple — pocos minimos, y el warm restart con lr alto destruye la estructura aprendida sin encontrar alternativas mejores. SGDR con lr reducido o sin SGDR es preferible.

### 4.3 Modelo 7M — Run v3 (67 epochs, cosine schedule)

**Tabla completa epoch-by-epoch:**

| Epoch | train_bpb | test_bpb | lr (relativo) | Delta_test/epoch |
|-------|-----------|----------|---------------|------------------|
| 1 | 3.105 | 2.174 | 0.298 | — |
| 2 | 2.056 | 1.641 | 0.596 | -0.533 |
| 3 | 1.754 | 1.451 | 0.891 | -0.190 |
| 4 | 1.609 | 1.357 | 0.991 | -0.094 |
| 5 | 1.531 | 1.305 | 0.986 | -0.052 |
| 10 | 1.384 | 1.199 | 0.946 | -0.021 |
| 15 | 1.331 | 1.161 | 0.881 | -0.008 |
| 20 | 1.305 | 1.142 | 0.796 | -0.004 |
| 25 | 1.290 | 1.129 | 0.694 | -0.003 |
| 30 | 1.280 | 1.121 | 0.582 | -0.002 |
| 35 | 1.273 | 1.115 | 0.465 | -0.001 |
| 40 | 1.266 | 1.108 | 0.350 | -0.001 |
| 45 | 1.260 | 1.107 | 0.243 | -0.000 |
| 50 | 1.254 | 1.101 | 0.151 | -0.001 |
| 55 | 1.249 | 1.096 | 0.077 | -0.001 |
| 60 | 1.246 | 1.094 | 0.027 | -0.000 |
| 65 | 1.245 | 1.092 | 0.002 | -0.000 |
| 67 | 1.244 | 1.092 | 0.000 | -0.000 |

**Duracion total**: 52h 49min, ~47 min/epoch, ~20k tok/s.

**VRAM peak**: 3.8 GB (con gradient checkpointing).

### 4.4 Analisis de convergencia — Modelo 7M

#### Fase 1: Descenso rapido (epochs 1-5)

```
Delta_test = 2.174 - 1.305 = 0.869 BPB en 5 epochs
Tasa media: -0.174 BPB/epoch
```

Esta fase corresponde al aprendizaje de estadisticas unigram y bigram del corpus. La entropia empirica de Python a nivel de byte para distribuciones unigram es ~4.5-5.0 BPB; para bigram ~2.0-2.5 BPB. El modelo alcanza 1.305 en epoch 5, ya por debajo del nivel bigram, indicando captura de dependencias de orden superior.

#### Fase 2: Refinamiento (epochs 5-30)

```
Delta_test = 1.305 - 1.121 = 0.184 BPB en 25 epochs
Tasa media: -0.007 BPB/epoch
```

Decaimiento exponencial: test_bpb(e) ≈ 1.10 + 0.20 * exp(-(e-5)/8).

#### Fase 3: Convergencia final (epochs 30-67)

```
Delta_test = 1.121 - 1.092 = 0.029 BPB en 37 epochs
Tasa media: -0.0008 BPB/epoch
```

El modelo esta en regimen de rendimientos decrecientes. La mejora marginal de 0.029 BPB en 37 epochs (29h de compute) sugiere proximidad al limite de capacidad del modelo para este corpus.

#### Gap de generalizacion

```
Gap = train_bpb - test_bpb

Epoch 1:  3.105 - 2.174 = 0.931  (test mejor que train — regularizacion dominante)
Epoch 10: 1.384 - 1.199 = 0.185
Epoch 30: 1.280 - 1.121 = 0.159
Epoch 50: 1.254 - 1.101 = 0.153
Epoch 67: 1.244 - 1.092 = 0.152
```

El gap se estabiliza en ~0.15 BPB a partir de epoch 10. No hay evidencia de overfitting: el gap no crece en el tiempo. Esto es consistente con:
1. Corpus suficientemente grande (56.6 MB >> parametros del modelo)
2. Regularizacion implicita de EntropicAdam (reduce LR en direcciones ruidosas)
3. bf16 como regularizador de precision

Nota: train_bpb > test_bpb puede parecer contraintuitivo. Esto ocurre porque train_bpb se calcula como promedio sobre un epoch completo (incluyendo pasos con LR alto al inicio), mientras test_bpb se evalua al final del epoch con los pesos actualizados. El modelo al final del epoch es estrictamente mejor que el modelo promedio durante el epoch.

---

## 5. Comparacion entre Modelos

### 5.1 Modelo 3.5M (d=512) vs 7M (d=1024)

Comparacion a iso-epochs en cosine schedule:

| Metrica | 3.5M (epoch 30) | 7M (epoch 30) | 7M (epoch 67) |
|---------|-----------------|---------------|---------------|
| test_bpb | 1.046 | 1.121 | 1.092 |
| test_loss | 0.725 | 0.777 | 0.757 |
| train_bpb | 1.185 | 1.280 | 1.244 |
| Throughput | ~55k tok/s | ~20k tok/s | ~20k tok/s |
| VRAM peak | ~10 GB | ~3.8 GB | ~3.8 GB |

**Observacion critica**: El modelo 3.5M (epoch 50, BPB=0.963) supera al 7M v3 (epoch 67, BPB=1.092) por un margen significativo de 0.129 BPB. Ambos modelos fueron entrenados con selective scan activo (delta_fast_mod y delta_slow_mod presentes y entrenados desde epoch 1).

### 5.2 Factores que explican la diferencia

Verificacion posterior (2026-09-21) confirmo que el 7M v3 SI fue entrenado con selective scan desde el inicio. Los parametros delta_*_mod tienen normas entre 9 y 22 en el checkpoint epoch 67, con valores en rango [-3.7, +2.7] — lejos de los zeros iniciales. La diferencia de rendimiento no se explica por presencia/ausencia de selective scan.

Hipotesis principal: **insuficiencia de datos para el modelo mas grande**.

1. **Ratio datos/parametros**: El corpus de ~60 MB contiene ~60M bytes. Con T=256 y 67 epochs, el modelo ve ~3.8 Gtok. Para 7M params esto da ~540 tokens/param. Para 3.5M params da ~1080 tokens/param. El modelo 7M tiene la mitad de tokens por parametro, lo que limita su capacidad de generalizacion.

2. **Gap train-test**: El 7M muestra train_bpb=1.244 vs test_bpb=1.092 (gap=0.152). El 3.5M muestra train_bpb=1.185 vs test_bpb=0.963 (gap=0.222 en baseline, reducido con selective scan). El gap menor del 7M no indica mejor generalizacion sino que ambos modelos estan limitados por los datos.

3. **Convergencia temprana**: El 7M muestra estancamiento desde epoch ~35 (Delta_test < 0.001/epoch). Con mas datos, la curva podria continuar descendiendo.

4. **VRAM anomalo**: El 7M reporta 3.8 GB peak con d=1024, mientras el 3.5M reporta 10 GB con d=512. Esto sugiere configuraciones diferentes de gradient checkpointing o batch effective size.

### 5.3 Scaling law empirico

Con solo dos puntos de datos, la extrapolacion es limitada, pero para referencia:

```
Si: BPB(N) = A * N^(-alpha) + C

Usando:
  BPB(3.5M) = 0.963  [selective scan, 50 epochs, ~2.8 Gtok]
  BPB(7.0M) = 1.092  [selective scan, 67 epochs, ~3.8 Gtok]
```

El modelo mas grande da peor resultado, lo que viola la relacion de scaling esperada. Dado que ambos modelos usan selective scan, la causa mas probable es la insuficiencia de datos: 60 MB de corpus no permite al modelo 7M aprovechar su capacidad adicional. Para derivar scaling laws validas seria necesario:
- Corpus significativamente mayor (>500 MB)
- Mismo numero de tokens vistos por parametro
- Mismo schedule y LR

---

## 6. Learning Rate Schedule

### 6.1 Cosine Annealing

```
factor(e) = warmup(e) * 0.5 * (1 + cos(pi * progress(e)))

warmup(e) = min(e / (warmup_frac * total_epochs), 1.0)
progress(e) = min((e - start_epoch) / (total_epochs - start_epoch), 1.0)

lr_effective(e) = lr_base * factor(e)
```

Para el run 7M v3 (67 epochs, warmup 5%):
- Epochs 0-3: warmup lineal 0 -> 1.0
- Epoch 3-4: lr maximo (~3e-4)
- Epochs 4-67: decaimiento cosine suave hasta 0

### 6.2 SGDR (Warm Restart Cosine)

Periodo inicial T_0 = max(total_epochs // 5, 50), duplicacion exponencial:

```
Ciclo 1: epochs [start, start + T_0)
Ciclo 2: epochs [start + T_0, start + 3*T_0)
Ciclo 3: epochs [start + 3*T_0, start + 7*T_0)
```

Para 50 epochs con T_0=10: ciclos de longitud 10, 20, 40 epochs.

### 6.3 Interaccion LR-schedule x EntropicAdam

El LR efectivo para el parametro i en el grupo j es:

```
lr_eff(i, j, e, t) = lr_base * bias_correction(t) * schedule_factor(e) * exp(-H(p_j) / T(e))
```

Donde:
- `bias_correction(t)` = sqrt(1 - beta2^t) / (1 - beta1^t), monotonamente creciente, satura en ~1.0 para t >> 1/(1-beta1)
- `schedule_factor(e)` = cosine o SGDR
- `exp(-H/T)` = modulacion entropica, rango [exp(-ln2/T), 1.0]

A T=0.5 (final del training), el rango de modulacion es [exp(-ln2/0.5), 1] = [0.25, 1.0], un factor 4x entre parametros mas y menos coherentes.

---

## 7. Analisis de la Funcion de Loss

### 7.1 Cross-entropy byte-level

```
L = -1/(B*T) * sum_{b,t} log P(x_{b,t+1} | x_{b,<=t})
```

donde P se calcula via softmax sobre 256 logits.

### 7.2 Cotas teoricas

**Cota superior trivial**: log(256) = 5.545 nats = 8.0 BPB (prediccion uniforme).

**Entropia empirica de Python** (estimaciones de la literatura y nuestros datos):
- Unigram: ~4.5-5.0 BPB
- Bigram: ~2.0-2.5 BPB
- Modelos de orden n (n=5-8): ~1.2-1.5 BPB
- Shannon entropy rate (limite): ~0.3-0.5 BPB para texto natural en ingles

**Cota inferior practica para modelos de 3-7M params sobre Python**:

La redundancia de Python (keywords, indentacion, patrones sintacticos) sugiere un floor mas bajo que ingles natural. Sin embargo, la capacidad limitada del modelo impone:

```
BPB_floor(3.5M) ≈ 0.85-0.95   (estimado)
BPB_floor(7M)   ≈ 0.75-0.85   (estimado)
```

Nuestro mejor resultado (0.963 BPB, 3.5M) esta dentro o cerca de este rango estimado.

### 7.3 Tasa de convergencia

Definimos la eficiencia de convergencia como:

```
eta = (BPB_init - BPB_final) / (tokens_procesados / 10^9)

Para 7M v3:
  BPB_init = 2.174
  BPB_final = 1.092
  tokens = 67 epochs * 56.6M bytes ≈ 3.79 * 10^9
  eta = (2.174 - 1.092) / 3.79 = 0.286 BPB/Gtok

Para 3.5M (runs 1-3, total 50 epochs):
  BPB_init ≈ 2.07
  BPB_final = 0.963
  tokens = 50 * 60M ≈ 3.0 * 10^9
  eta = (2.07 - 0.963) / 3.0 = 0.369 BPB/Gtok
```

El modelo 3.5M tiene mejor eficiencia de convergencia por token, consistente con que modelos mas pequenos convergen mas rapido (en tokens) pero a floors mas altos.

---

## 8. Analisis del Selective Scan

### 8.1 Impacto cuantitativo

El cambio de decay constante a content-dependent en el modelo 3.5M:

| Metrica | Decay constante (epoch 30) | Selective scan (epoch 50) | Mejora |
|---------|---------------------------|--------------------------|--------|
| test_bpb | 1.046 | 0.963 | -0.083 (-7.9%) |
| test_loss | 0.725 | 0.668 | -0.057 (-7.9%) |
| Convergencia/epoch | 0.003 BPB/epoch | 0.016 BPB/epoch | 5.3x |

### 8.2 Mecanismo

Con decay constante:
```
lambda_fast[dim_k] = exp(-softplus(delta_fast[k]))    (fijo por dimension)
```

Con selective scan:
```
lambda_fast[b,t,k] = exp(-softplus(delta_fast[k] + state[b,t,k] * delta_fast_mod[k]))
```

La modulacion `state * delta_fast_mod` permite que el decay varie segun el contenido del token actual. En la practica esto permite:

- **Apertura de parentesis/llaves**: delta aumenta (decay bajo = retener contexto)
- **Cierre de parentesis/llaves**: delta disminuye (decay alto = olvidar contexto local)
- **Whitespace/indentacion**: patrones intermedios aprendidos por el modelo

### 8.3 Inicializacion y continuidad

`delta_fast_mod` y `delta_slow_mod` se inicializan a cero. Esto garantiza:

```
lambda_t(init) = exp(-softplus(delta_base + state * 0)) = exp(-softplus(delta_base)) = lambda_original
```

El comportamiento es identico al modelo pre-cambio en el paso 0, permitiendo continuar desde un checkpoint sin discontinuidad.

---

## 9. Throughput y Eficiencia Computacional

### 9.1 Comparacion de implementaciones

| Implementacion | tok/s | s/batch | Speedup | Notas |
|---------------|-------|---------|---------|-------|
| PyTorch WHT + scan | ~20k | ~0.81 | 1.0x | Baseline |
| CUDA fused (decay cte) | ~55k | ~0.30 | 2.75x | Kernels JIT sm_89 |
| CUDA selective scan | ~55k | ~0.52 | 1.56x | Variable decay kernel |

### 9.2 VRAM

| Config | d | L | B | T | Precision | Grad ckpt | Peak VRAM |
|--------|---|---|---|---|-----------|-----------|-----------|
| 3.5M | 512 | 12 | 64 | 256 | bf16 | Si | ~6.8 GB |
| 3.5M | 512 | 12 | 64 | 256 | bf16 | Si + fused | ~10 GB |
| 7M v3 | 1024 | 12 | 64 | 256 | bf16 | Si | ~3.8 GB |

Nota: La diferencia de VRAM entre 3.5M (10 GB) y 7M (3.8 GB) es contraintuitiva. La explicacion probable es que las activaciones intermedias de los kernels CUDA fused consumen mas VRAM que la implementacion PyTorch con gradient checkpointing, y el 7M v3 puede no estar usando los kernels fused (d=1024 > 1024 threshold is borderline).

### 9.3 Tiempo de entrenamiento

| Run | Epochs | min/epoch | Total | GPU |
|-----|--------|-----------|-------|-----|
| 3.5M Run 1 | 30 | ~16 | ~8h | RTX 4090 |
| 3.5M Run 3 | 5 | ~17 | ~1.5h | RTX 4090 |
| 7M v3 | 67 | ~47 | ~52.8h | RTX 4090 |

---

## 10. Resumen de Hallazgos

### 10.1 Resultados principales

1. **Mejor resultado absoluto**: test_bpb = 0.963 (modelo 3.5M, epoch 50, selective scan + cosine schedule). Esto equivale a una compresion de 8.3:1 sobre Python source code a nivel de byte.

2. **Selective scan vs. decay fijo**: Mejora de 7.9% en BPB y 5.3x en velocidad de convergencia (BPB/epoch). El overhead computacional es compensado por menos epochs necesarios.

3. **SGDR no es efectivo post-convergencia**: En modelos de 3.5M params, el warm restart con lr=3e-4 destruye estructura sin encontrar minimos mejores. La degradacion es monotona e irreversible.

4. **El modelo 7M v3 no supera al 3.5M**: test_bpb 1.092 vs 0.963. Ambos usan selective scan. La causa probable es insuficiencia de datos: el corpus de 60 MB no provee suficientes tokens/parametro para que el 7M aproveche su capacidad extra (540 tok/param vs 1080 tok/param del 3.5M).

### 10.2 Limitaciones del analisis

- Solo dos tamanios de modelo evaluados; no se pueden derivar scaling laws.
- Un solo corpus (Python). Generalizacion a otros dominios no evaluada.
- El corpus de 60 MB es insuficiente para evaluar scaling: el 7M tiene ~540 tokens/param vs ~1080 del 3.5M.
- No se midio la calidad generativa (perplexity en generacion, coherencia semantica).
- Hiperparametros no fueron optimizados sistematicamente (no grid search, no Bayesian optimization).

### 10.3 Corpus disponibles para escalamiento

Verificacion (2026-09-21) de los corpus presentes en la instancia vast.ai:

| Corpus | Tamanio | Train chunks | Batches/epoch (B=64) | tok/param (7M, 67ep) | Tiempo/epoch est. |
|--------|---------|-------------|---------------------|---------------------|-------------------|
| corpus_python.txt | 60 MB | 221,248 | 3,457 | 540 | ~47 min |
| corpus_python_200mb.txt | 200 MB | 737,280 | 11,520 | 1,796 | ~157 min |
| corpus_python_1gb.txt | 536 MB | 1,974,952 | 30,858 | 4,811 | ~422 min |

Todos son codigo Python del mismo scope (repos open-source de GitHub).

**Analisis de viabilidad (RTX 4090, ~0.82s/batch):**

- **200 MB corpus, 67 epochs**: ~176h (~7.3 dias). Viable. tok/param = 1796, comparable al ratio del 3.5M con 60MB (1080). Deberia desbloquear capacidad del 7M.
- **536 MB corpus, 67 epochs**: ~471h (~20 dias). Inviable con 67 epochs. Con 20 epochs: ~140h (~5.8 dias), tok/param = 1436. Viable.
- **200 MB corpus, 30 epochs**: ~78h (~3.3 dias). tok/param = 804. Compromiso razonable.

**Nota sobre resume vs from-scratch**: Resumir desde el checkpoint epoch 67 (entrenado en 60MB) conserva patrones generales de Python ya aprendidos. Con un corpus mas grande, el test_bpb inicial subira temporalmente (distribucion ligeramente distinta) pero la convergencia sera mas rapida que entrenar desde cero. El floor de convergencia deberia ser inferior a 1.092 gracias a la mayor diversidad de datos.

**Nota sobre kernels CUDA**: No se requieren cambios. Los kernels operan por batch (B, T, d) independientemente del tamanio del corpus. WHT (d=1024 <= 1024 max threads) y selective scan (T=256 <= 1024) permanecen dentro de limites hardware. Solo cambia el numero de batches por epoch.

### 10.4 Proximos pasos experimentales sugeridos

1. **Entrenar 7M con corpus 200MB** (resume desde epoch 67, lr=1e-4, 30 epochs, ~3.3 dias). Primer test de scaling con datos suficientes.
2. **Evaluar generacion** en CPU con el checkpoint 3.5M epoch 50.
3. **Comparar BPB**: 7M-60MB vs 7M-200MB vs 3.5M-60MB para aislar efecto de datos vs parametros.
4. **Si 200MB mejora**: considerar corpus 536MB con epochs reducidos (20 epochs, ~5.8 dias).
5. **Sweep de learning rate**: {1e-4, 2e-4, 3e-4, 5e-4} con cosine schedule sobre 200MB.
