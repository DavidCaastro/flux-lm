# Flux LM: Fundamentos Tecnicos, Evolucion e Innovaciones

> Documento de referencia tecnica. Analiza que elementos de Flux son prestados
> de la literatura, cuales son combinaciones originales, y cuales son innovaciones
> genuinas. Sin adjetivos grandilocuentes — solo hechos verificables.
>
> Revision: 2026-09-21

---

## Linea Temporal

```
1867  Sylvester          Matrices de Hadamard (construccion recursiva)
1923  Walsh              Funciones de Walsh (base ortonormal {+1,-1})
1965  Varios             Algoritmo butterfly para WHT — O(n log n)
                         (analogo a Cooley-Tukey FFT, publicado el mismo ano)
1986  Hillis & Steele    Parallel prefix scan — O(n log n) work, O(log n) depth
1990  Blelloch           Work-efficient parallel scan — O(n) work, O(log n) depth
1997  Hochreiter &       LSTM — gated recurrence con forget/input/output gates
      Schmidhuber
2014  Cho et al.         GRU — recurrence simplificada con 2 gates
2016  Moczulski et al.   ACDC — Hadamard + diagonales para compresion de redes
2017  Vaswani et al.     Transformer — self-attention O(T^2 d)
2019  Zhang & Sennrich   RMSNorm
2019  Dao et al.         Butterfly matrices — factorizaciones aprendibles O(n log n)
2020  Gu et al.          HiPPO / S4 — State Space Models continuos
2021  Lee-Thorp et al.   FNet — FFT reemplaza attention (token mixing)
2023  Gu & Dao           Mamba — selective state space model (content-dependent)
2024  Dao & Gu           Mamba-2, Jamba, y variantes SSM
2026  Este trabajo       Flux LM — WHT spectral gating + dual-timescale
                         content-dependent recurrence + SPM
```

---

## 1. Fundamentos Matematicos

### 1.1 La Transformada Walsh-Hadamard (WHT)

La WHT de un vector x in R^d (donde d = 2^n) se define como:

```
y = H_d * x
```

donde H_d es la matriz de Hadamard de orden d, construida recursivamente:

```
H_1 = [1]

H_{2k} = [ H_k    H_k  ]
          [ H_k   -H_k  ]
```

Propiedades fundamentales:

1. **Ortogonalidad**: H_d * H_d^T = d * I_d
2. **Involucion**: H_d * H_d = d * I (la WHT es su propia inversa, salvo escala)
3. **Entradas**: todos los elementos son +1 o -1
4. **Normalizacion**: con el factor 1/sqrt(d), la transformada es unitaria:
   (H_d/sqrt(d))^2 = I

La WHT descompone un vector en sus componentes de "sequency" — el analogo Walsh
de las frecuencias en Fourier. La sequency k mide el numero de cambios de signo
en la k-esima fila de H_d.

### 1.2 Fast Walsh-Hadamard Transform (FWHT)

La FWHT es el algoritmo O(d log d) para computar H_d * x, analogo a la FFT para
la DFT. Explota la estructura recursiva de H_d mediante operaciones butterfly:

```
Para s = 0, 1, ..., log2(d) - 1:
    half = 2^s
    Para cada par (i, i+half) en grupos de 2*half:
        a = x[i]
        b = x[i + half]
        x[i]      = a + b
        x[i+half] = a - b
x = x / sqrt(d)
```

**Complejidad**: d/2 * log2(d) sumas, 0 multiplicaciones (excepto la normalizacion final).

Para d = 512: 9 etapas butterfly, 2304 operaciones de suma.
Para d = 1024: 10 etapas butterfly, 5120 operaciones de suma.

### 1.3 WHT vs FWHT en Flux: son lo mismo

En el codigo de Flux, `wht()` implementa el algoritmo butterfly — es decir, ES la FWHT:

```python
# model.py, lineas 66-80
def wht(x):
    half = 1
    while half < d:                              # log2(d) iteraciones
        x = x.view(-1, d // (2*half), 2, half)   # agrupar pares
        a = x[:, :, 0, :]                        # mitad superior
        b = x[:, :, 1, :]                        # mitad inferior
        x = stack([a + b, a - b], dim=2)          # butterfly
        half *= 2
    return x * (1/sqrt(d))                        # normalizacion unitaria
```

El kernel CUDA (`wht_kernel` en `kernels.py`) implementa el mismo butterfly en shared
memory, con un thread por elemento:

```c
// kernels.py, lineas 38-70
for (int s = 0; s < log_d; s++) {
    int half = 1 << s;
    // ... butterfly con __syncthreads() entre etapas
    smem[tid] = (idx < half) ? (a + b) : (a - b);
}
out[row*d + tid] = smem[tid] * rsqrtf(d);
```

**No hay una version "lenta" (WHT naive O(d^2)) en el codigo.** Toda referencia a "WHT"
en Flux se refiere a la implementacion rapida O(d log d). El nombre `wht()` es una
convencion — la T de "Transform" no implica velocidad, solo la operacion matematica.

### 1.4 Parallel Scan (Prefix Sum con Recurrencia Lineal)

El parallel scan computa la recurrencia lineal:

```
y[0] = x[0]
y[t] = lambda * y[t-1] + x[t],    t = 1, ..., T-1
```

**Decay constante** (lambda escalar por dimension):

El algoritmo de Hillis-Steele explota que la recurrencia puede reescribirse como un
prefix sum con operador asociativo. Para decay constante, las potencias cuadradas de
lambda aceleran el proceso:

```
stride = 1, decay_pow = lambda
while stride < T:
    y[t] += decay_pow * y[t - stride]    (para t >= stride)
    decay_pow = decay_pow^2
    stride *= 2
```

Complejidad: O(T log T) work, O(log T) depth. Implementado en CUDA con un thread
por timestep y shared memory.

### 1.5 Associative Scan Generalizado (Decay Variable)

Cuando lambda varia por timestep — lambda[t] depende del contenido — la tecnica de
potencias cuadradas no aplica. Se usa el operador asociativo general:

```
(a1, b1) + (a2, b2) = (a2 * a1, a2 * b1 + b2)
```

donde a = decay, b = valor acumulado. Este operador es asociativo (verificable
por expansion directa) pero NO conmutativo.

```python
# model.py, lineas 106-115
a = decay    # (B, T, d) — variable por timestep
b = x        # (B, T, d)
stride = 1
while stride < T:
    a_prev = pad(a[:, :-stride], value=1.0)
    b_prev = pad(b[:, :-stride])
    b = a * b_prev + b       # b_{i} = a_{i} * b_{i-stride} + b_{i}
    a = a * a_prev            # a_{i} = a_{i} * a_{i-stride}
    stride *= 2
```

El kernel CUDA (`variable_scan_kernel` en `selective_scan_kernel.py`) implementa
esto en shared memory con dos buffers (sa, sb) y log2(T) sincronizaciones.

**Backward del scan variable**: la derivada requiere un scan reverso con decay
desplazado. Sea g = grad_output, grad_x se obtiene via:

```
grad_x[T-1] = g[T-1]
grad_x[t] = decay[t+1] * grad_x[t+1] + g[t]
```

Nota critica: usa `decay[t+1]` (timestep SIGUIENTE), no `decay[t]`. Este shift
fue un bug corregido (commit 995e7f1) — sin el, los gradientes de los parametros
de decay son incorrectos.

---

## 2. Arquitectura Flux: Formulacion Completa

### 2.1 Diagrama de flujo

```
Entrada: byte_ids (B, T) in {0..255}

    |
    v
[Embedding]  x = Emb(byte_ids)              (B,T) -> (B,T,d)
    |
    v
[FluxLayer x L]  (detallado abajo)
    |
    v
[Linear Head]  logits = x @ W_head^T + b     (B,T,d) -> (B,T,256)
    |
    v
[CrossEntropy]  loss = CE(logits, targets)
```

### 2.2 FluxLayer: forward pass completo

Para cada capa l in {0, ..., L-1}, con input x de forma (B, T, d):

**Paso 1 — RMSNorm:**
```
rms = sqrt( mean(x^2, dim=-1) + 1e-8 )
state = (x / rms) * gamma                    gamma in R^d, init = ones
```

**Paso 2 — Gating (byte-conditioned + state-feedback):**
```
gate = sigma( g_gate[byte_id] + w_gate_h * h_fast_{prev} )
state = gate * state + a_bias[byte_id]
```
- g_gate: Embedding(256, d), init N(0, 0.1)
- a_bias: Embedding(256, d), init diagonal 0.3
- w_gate_h: R^d, init N(0, 0.1)
- En modo paralelo (pass 0): h_fast_{prev} = 0; (pass 1+): h_fast desplazado

**Paso 3 — WHT spectral gating (x2):**
```
state = tanh( s1 * WHT(state) + b1 )         s1, b1 in R^d
state = tanh( s2 * WHT(state) + b2 )         s2, b2 in R^d
```
- s1, s2: init = ones (identidad en dominio espectral)
- b1, b2: init = zeros (sin sesgo)
- WHT = FWHT butterfly O(d log d)
- Kernel fused: `wht_scale_tanh_fused` combina WHT + scale + bias + tanh en 1 launch

**Paso 4 — Content-dependent decay (selective scan):**
```
delta_fast_t = delta_fast + state * delta_fast_mod
lambda_fast_t = exp( -softplus(delta_fast_t) )

delta_slow_t = delta_slow + state * delta_slow_mod
lambda_slow_t = exp( -softplus(delta_slow_t) )
```
- delta_fast: R^d, init linspace(0.1, 0.5)  ->  lambda_fast in (0.38, 0.60)
- delta_slow: R^d, init linspace(-2.0, 0.5)  ->  lambda_slow in (0.60, 0.95)
- delta_fast_mod, delta_slow_mod: R^d, init = zeros (sin modulacion inicial)
- softplus(x) = log(1 + exp(x)) garantiza positividad
- exp(-softplus(x)) in (0, 1) garantiza estabilidad del scan

**Paso 5 — Parallel scan (dual-timescale):**
```
h_fast = scan(lambda_fast_t, b_in_fast * state)     track rapido
h_slow = scan(lambda_slow_t, b_in_slow * state)     track lento
```
- b_in_fast, b_in_slow: R^d, init = ones
- scan variable: O(T d log T) via operador asociativo
- Cuando delta_*_mod = 0: decay es constante, equivalente a scan clasico

**Paso 6 — SPM (Slow Persistent Memory):**
```
z = h_slow[::STRIDE] @ W_spm^T               (B, T/4, K)    K=4
h_sem = scan(lam_sem, (1 - lam_sem) * z)      filtro IIR pasa-bajos
cond = (h_sem @ W_spm) * sigma(spm_gate)      back to R^d
cond = repeat_interleave(cond, STRIDE)[:T]     upsample
```
- W_spm: R^{K x d}, init Xavier
- spm_delta: R^K, init linspace(-5, -3) -> lam_sem in (0.95, 0.99)
- spm_gate: R^d, init = -5.0 (sigma(-5) ~ 0.007, casi apagado)
- STRIDE = 4: opera a 1/4 de resolucion temporal
- Usa scan con decay CONSTANTE (kernel fused de `kernels.py`)

**Paso 7 — Output (dynamic gating + skip + residual + conditioning):**
```
c_f = sigma( c_out_fast + w_c_fast * state )     gating fast
c_s = sigma( c_out_slow + w_c_slow * state )     gating slow
out = c_f * h_fast + c_s * h_slow + skip * state + res_scale * x + cond
```
- c_out_fast, c_out_slow: R^d, init = -1.0 -> sigma(-1) ~ 0.27
- w_c_fast, w_c_slow: R^d, init = zeros (gating inicialmente estatico)
- skip: R^d, init = 0.5
- res_scale = 1/log(l+2): decae con profundidad (1.44 -> 0.39)

### 2.3 Parametros por capa (d=512)

| Componente | Parametros | Formula |
|-----------|-----------|---------|
| RMSNorm gamma | 512 | d |
| g_gate (Emb) | 131,072 | 256d |
| a_bias (Emb) | 131,072 | 256d |
| w_gate_h | 512 | d |
| WHT blocks (s1,b1,s2,b2) | 2,048 | 4d |
| Decay (delta_fast/slow + mods) | 2,048 | 4d |
| Scan inputs (b_in_fast/slow) | 1,024 | 2d |
| Output gates (c_out_fast/slow, w_c_fast/slow) | 2,048 | 4d |
| skip | 512 | d |
| SPM (spm_delta, spm_gate) | 516 | K + d |
| **Total por capa** | **~271,000** | **~529d + K** |

Nota: el ~97% de los parametros por capa estan en las dos embeddings de gating
(g_gate, a_bias). El WHT en si tiene 0 parametros.

---

## 3. Trabajo Relacionado y Comparacion

### 3.1 Transformer (Vaswani et al., 2017)

```
Attention: Q,K,V = linear(x), attn = softmax(QK^T/sqrt(d_k)) V
MLP:       h = GELU(x W1) W2
```

| Aspecto | Transformer | Flux |
|---------|------------|------|
| Token mixing | Self-attention O(T^2 d) | Parallel scan O(T d log T) |
| Feature mixing | MLP O(T d^2) | WHT O(T d log d) |
| Params/capa (d=512) | ~2M (QKV + MLP 4d) | ~271K |
| Causal masking | Mascara triangular | Implicita (scan es causal) |
| Contexto | T^2 memoria | T lineal (scan) |
| Inference | O(T) por token nuevo (KV cache) | O(1) por token (estado recurrente) |

### 3.2 FNet (Lee-Thorp et al., 2021)

FNet reemplaza self-attention con FFT:
```
x = FFT_seq(FFT_feature(x)).real
```

| Aspecto | FNet | Flux |
|---------|------|------|
| Transformada | FFT (compleja) | WHT (real) |
| Eje de mixing | Secuencia Y features | Solo features (scan para secuencia) |
| No-linealidad | Ninguna en la capa FFT | tanh(s*WHT+b) — gating espectral |
| Causalidad | NO (FFT sobre toda la secuencia) | SI (scan es causal) |
| Aprendibilidad | 0 params en FFT | 4d params (s1,b1,s2,b2) por WHT |

Diferencia clave: FNet aplica FFT en AMBOS ejes (token y feature) sin causalidad.
Flux aplica WHT solo en el eje de features y usa scan causal para el eje temporal.
FNet no puede hacer generacion autoregresiva sin modificaciones.

### 3.3 S4 / SSMs (Gu et al., 2020-2022)

S4 discretiza un sistema lineal continuo:
```
dx/dt = A x + B u
y = C x + D u
```

Con discretizacion ZOH y diagonalizacion de A:

```
x_t = A_bar x_{t-1} + B_bar u_t
y_t = C x_t
```

| Aspecto | S4 | Flux |
|---------|-----|------|
| Formulacion | SSM continuo discretizado | Recurrencia directa discreta |
| Matriz A | HiPPO (estructura especifica) | Diagonal (decay por dim) |
| Dimensionalidad | Estado N por feature | Estado 1 por feature (h_fast, h_slow) |
| Feature mixing | Ninguno (per-feature) | WHT (global, O(d log d)) |
| Parametrizacion | A, B, C, D | delta, b_in, c_out (directo) |

### 3.4 Mamba (Gu & Dao, 2023)

Mamba anade selectividad al SSM:
```
B_t = Linear_B(x_t)        # input projection, variable
C_t = Linear_C(x_t)        # output projection, variable
Delta_t = softplus(Linear_Delta(x_t))   # discretization step
A_bar_t = exp(Delta_t * A)
x_t = A_bar_t * x_{t-1} + Delta_t * B_t * u_t
y_t = C_t * x_t
```

| Aspecto | Mamba | Flux |
|---------|-------|------|
| Decay variable | Si: exp(Delta * A) | Si: exp(-softplus(delta + state * mod)) |
| Param. del decay | W_Delta (d_inner x d) — projeccion lineal | delta_mod (d,) — modulacion element-wise |
| Input projection | B_t = Linear(x), variable | b_in: constante (ones) |
| Output projection | C_t = Linear(x), variable | c_out + w_c * state: semi-variable |
| Feature mixing | Conv1D + Linear projs | WHT (0 params para mixing) |
| Estado | N valores por feature | 2 valores por feature (fast + slow) |
| SPM | No | Si (filtro pasa-bajos strided) |
| Scan impl. | Hardware-aware (Dao) | Hillis-Steele en shared memory |
| Params/capa (d=512) | ~1.5M-3M | ~271K |

Diferencia clave en la parametrizacion del decay:
- Mamba: `Delta = softplus(W * x + b)` con W in R^{d x d} — O(d^2) parametros
- Flux: `delta_t = delta_base + state * delta_mod` — O(d) parametros (element-wise)

Flux sacrifica expresividad (sin cross-feature interactions en el decay) a cambio de
~1000x menos parametros en la modulacion del decay. La WHT previa compensa parcialmente
al mezclar features antes de computar el decay.

### 3.5 RWKV (Peng et al., 2023)

```
r_t = sigma(W_r * x_t)
k_t = W_k * x_t
v_t = W_v * x_t
wkv_t = (sum_i exp(w*i + k_i) * v_i) / (sum_i exp(w*i + k_i))
o_t = W_o * (r_t * wkv_t)
```

| Aspecto | RWKV | Flux |
|---------|------|------|
| Recurrence | Weighted sum exponencial | Parallel scan lineal |
| Decay | Decay posicional (w) | Content-dependent |
| Feature mixing | Linear projections O(d^2) | WHT O(d log d) |
| Gates | r (receptance) | g_gate + c_out dynamic |

### 3.6 Butterfly Matrices (Dao et al., 2019)

Dao et al. propusieron matrices butterfly APRENDIBLES como reemplazo de capas
lineales densas:

```
B = prod_{i=1}^{log(d)} B_i
```

donde cada B_i es una matriz butterfly sparse con O(d) parametros.

| Aspecto | Butterfly aprendible | Flux WHT |
|---------|---------------------|----------|
| Estructura | Butterfly con pesos aprendidos | Butterfly FIJO (Hadamard) |
| Parametros de mixing | O(d log d) | 0 |
| Parametros de control | Dentro de la matriz | Externos: s, b (2d por bloque) |
| Expresividad del mixing | Alta (aprende la transformada) | Fija (siempre WHT) |
| Simplicidad | Media | Alta |

Flux toma la decision opuesta: fijar la estructura butterfly como WHT y controlar
la respuesta espectral con parametros FUERA de la transformada (scale, bias, tanh).
Esto reduce parametros pero limita el espacio de transformadas a:
tanh(s * H_d * x / sqrt(d) + b), un subconjunto de las transformadas generales.

---

## 4. Analisis de Innovacion

### 4.1 Elementos prestados (con atribucion)

| Elemento | Origen | Referencia |
|----------|--------|-----------|
| WHT / FWHT | Walsh 1923, algoritmo butterfly ~1965 | Beauchamp (1975) |
| Parallel scan | Hillis & Steele 1986, Blelloch 1990 | JACM |
| Gated recurrence | LSTM (Hochreiter 1997), GRU (Cho 2014) | Neural Computation |
| RMSNorm | Zhang & Sennrich 2019 | NeurIPS |
| Content-dependent selection | Mamba (Gu & Dao 2023) | concepto general |
| Residual connections | ResNet (He et al. 2015) | CVPR |
| Byte-level LM | Varios (Gillick et al. 2015, Al-Rfou et al. 2019) | — |
| Cosine annealing | Loshchilov & Hutter 2016 (SGDR) | ICLR |
| Embedding lookup como gate | Adaptado de Transformer embeddings | — |

### 4.2 Elementos originales o combinaciones nuevas

**4.2.1 WHT como mecanismo primario de feature mixing en un LM**

Hasta donde se puede verificar en la literatura, Flux es el primer modelo de
lenguaje que usa la WHT como unico mecanismo de mixing entre features, sin capas
lineales densas (MLP) ni attention.

Trabajo previo usa WHT/Hadamard para:
- Compresion de redes existentes (ACDC, Moczulski 2016)
- Random features / hashing (SimHash, FlyHash)
- Aproximacion de attention (H-Transformer)
- Butterfly aprendible como reemplazo de linear (Dao 2019)

Ninguno la usa como mixing PRIMARIO en un modelo recurrente para LM.

**4.2.2 Spectral gating: tanh(s * WHT(x) + b)**

La combinacion especifica:
1. WHT para mover a dominio de sequency
2. Scale element-wise (s) para amplificar/atenuar cada componente espectral
3. Bias (b) para desplazar
4. tanh para acotar y no-linealizar

Esto crea un **filtro espectral aprendible no-lineal** con solo 2d parametros.
Equivale a:

```
y_k = tanh(s_k * (H_d * x)_k + b_k)
```

Donde k indexa la componente de sequency. El modelo aprende que "frecuencias"
Walsh amplificar, cuales suprimir, y cuales invertir (via s_k negativo).

Dos bloques WHT consecutivos componen funciones mas complejas:
```
state = tanh(s2 * WHT(tanh(s1 * WHT(x) + b1)) + b2)
```

Esta composicion puede aproximar transformaciones no-lineales del espacio de
features con O(d log d) operaciones en vez de O(d^2) de una capa lineal densa.

**Limitacion**: la WHT es fija — no aprende QUE mezclar, solo COMO ponderar
la mezcla predeterminada. Para features que requieren interacciones especificas
no alineadas con la base Walsh, esto es suboptimo.

**4.2.3 Modulacion multiplicativa del decay (vs. proyeccion lineal)**

```
Mamba:  Delta = softplus(W_Delta @ x + b_Delta)      O(d^2) params
Flux:   delta_t = delta_base + state * delta_mod      O(d) params
```

La modulacion element-wise `state * delta_mod` es mas simple y barata que la
proyeccion lineal de Mamba. Funciona porque la WHT previa ya mezclo features
globalmente — `state` ya contiene informacion de todas las dimensiones.

Secuencia: WHT mezcla features -> state captura contexto global -> multiplicacion
element-wise modula decay -> scan propaga selectivamente.

**4.2.4 Dual-timescale con SPM strided**

La combinacion de tres escalas temporales en una sola capa:
- **Fast track**: decay ~0.4-0.6, captura patrones de 2-5 tokens
- **Slow track**: decay ~0.6-0.95, captura patrones de 5-20 tokens
- **SPM**: decay ~0.95-0.99, strided x4, captura tendencias de 20-100+ tokens

Cada track tiene su propia recurrencia independiente, y se combinan con gating
dinamico. El SPM opera en espacio comprimido (K=4 dims) a resolucion reducida.

Esta arquitectura multi-escala dentro de una sola capa no tiene precedente directo.
S4 usa un solo estado con multiples dimensiones; Mamba usa un solo track;
LSTM tiene un solo estado con forget gate.

**4.2.5 Dynamic output gating**

```
c_f = sigma(c_out_fast + w_c_fast * state)
c_s = sigma(c_out_slow + w_c_slow * state)
out = c_f * h_fast + c_s * h_slow + ...
```

El blending entre tracks rapido y lento depende del contenido actual. En tokens
que requieren memoria a corto plazo (e.g., cerrar un parentesis), c_f domina.
En tokens que requieren contexto largo (e.g., nombre de variable definida
hace 50 tokens), c_s domina.

**4.2.6 EntropicAdam**

Optimizador que modula el learning rate por grupo de parametros basandose en la
entropia binaria de la historia de signos del gradiente:

```
sign_history <<= 1; sign_history |= majority(grad > 0)
p = popcount(sign_history) / 32
H = -(p log(p) + (1-p) log(1-p))
glr = lr_base * exp(-H / T_epoch)
```

- H alta (signos aleatorios) -> LR bajo (conservador)
- H baja (signos consistentes) -> LR alto (agresivo)

No se ha encontrado esta formulacion especifica en la literatura de optimizacion.
La idea mas cercana es Rprop (solo signo) y variantes de Adam con adaptacion
de segundo orden, pero ninguna usa entropia de signos con temperatura decayente.

### 4.3 Tabla de Complejidad Comparativa

**FLOPs por capa, por token:**

| Operacion | Transformer | Mamba | FNet | Flux |
|-----------|------------|-------|------|------|
| Feature mixing | O(d^2) MLP | O(d * N) proj | O(d log d) FFT | O(d log d) WHT |
| Token mixing | O(T * d) attn | O(d) scan | O(T log T) FFT | O(d) scan |
| Total/token | O(d^2 + Td) | O(dN + d) | O(d log d + T log T) | O(d log d + d) |

Para d=512, T=256, N=16 (estado Mamba):

| Modelo | FLOPs/token/capa | Params/capa |
|--------|-----------------|-------------|
| Transformer | ~393K | ~2M |
| Mamba (d_inner=1024) | ~25K | ~1.5M |
| FNet | ~13K | ~0 (FFT) + MLP |
| **Flux** | **~5K** | **~271K** |

**FLOPs totales (L=12, T=256, forward completo):**

| Modelo | FLOPs | Params totales |
|--------|-------|---------------|
| Transformer (d=512) | ~1.2G | ~25M |
| Mamba (d=512, N=16) | ~80M | ~18M |
| Flux (d=512) | ~27M | ~3.5M |
| Flux (d=1024) | ~65M | ~7M |

Nota: Flux es ~45x mas eficiente en FLOPs que un Transformer del mismo d,
pero tiene ~7x menos parametros, lo que limita su capacidad.

---

## 5. Implementacion CUDA

### 5.1 WHT Fused Kernel

**Archivo**: `kernels.py`, funcion `wht_kernel`

Estrategia: un bloque CUDA por fila (token), un thread por dimension.

```
Block:    blockIdx.x = row index (0..N-1)
Threads:  threadIdx.x = dimension index (0..d-1)
Shared:   d * sizeof(float) bytes
```

Cada thread carga un elemento en shared memory, ejecuta log2(d) etapas butterfly
con `__syncthreads()` entre cada etapa, y escribe el resultado normalizado.

Limitacion: d <= 1024 (max threads per block en CUDA). Para d > 1024, fallback
a la implementacion PyTorch.

**Fused variant** (`wht_scale_tanh_kernel`): combina WHT + scale + bias + tanh
en un solo kernel, eliminando dos lecturas/escrituras intermedias a global memory.

**Backward**: WHT es su propia inversa (salvo escala), asi que:
```
grad_x = WHT(grad_output)     # misma operacion, reutiliza el kernel forward
```

Para el fused variant, el backward se computa analiticamente:
```
dtanh = 1 - tanh_out^2
grad_bias = sum(dtanh * grad, dim=0)
grad_scale = sum(dtanh * grad * wht_x, dim=0)
grad_x = WHT(scale * dtanh * grad)
```

### 5.2 Parallel Scan Kernel (Decay Constante)

**Archivo**: `kernels.py`, funcion `parallel_scan_kernel`

Layout: input pre-transpuesto a (B*D, T). Un bloque por fila B*D, un thread
por timestep.

```
Block:    blockIdx.x = bd index (0..B*D-1)
Threads:  threadIdx.x = timestep (0..T-1)
Shared:   T * sizeof(float) bytes
```

Hillis-Steele con `decay_pow = decay_pow^2` entre etapas.

Limitacion: T <= 1024. Para T > 1024, fallback automatico.

**Backward**: el scan reverso `grad_x[t] = decay * grad_x[t+1] + g[t]` se
implementa via flip + scan forward + flip (reutiliza el mismo kernel).

### 5.3 Selective Scan Kernel (Decay Variable)

**Archivo**: `selective_scan_kernel.py`, funcion `variable_scan_kernel`

Mismo layout (B*D, T), pero con dos buffers shared memory: sa (decay acumulado),
sb (valor acumulado).

```
Shared:   2 * T * sizeof(float) bytes
```

Kernel principal:
```c
for (stride = 1; stride < T; stride *= 2) {
    if (tid >= stride) {
        sb[tid] = a_cur * b_prev + b_cur;   // operador asociativo
        sa[tid] = a_cur * a_prev;
    }
}
```

Salida adicional: `a_out` (decay acumulado) para el backward, que necesita
el scan reverso con `decay[t+1]`.

Toda la aritmetica interna es float32 (independiente del dtype de entrada)
para evitar perdida de precision en los productos acumulados de decay.

---

## 6. EntropicAdam: Formulacion Completa

Ver seccion 2 de `docs/training_analysis.md` para la derivacion matematica
detallada. Resumen del algoritmo:

```
Input: params theta, lr, betas=(beta1,beta2), eps, T_init, T_final,
       total_epochs, group_size gs

Init: m=0, v=0, sign_history=0 (int64 per group of gs params), step=0

For each step:
    step += 1
    For each parameter p with gradient g:
        // Adam moments
        m = beta1 * m + (1-beta1) * g
        v = beta2 * v + (1-beta2) * g^2

        // Sign entropy per group
        For each group of gs contiguous params:
            majority = 1 if sum(g[group] > 0) >= gs/2 else 0
            sign_history = ((sign_history << 1) & 0xFFFFFFFF) | majority
            bits = popcount(sign_history)
            p_ratio = bits / 32
            H = -(p_ratio * log(p_ratio) + (1-p_ratio) * log(1-p_ratio))

        // Temperature decay
        frac = epoch / total_epochs
        T_epoch = T_init * (T_final / T_init)^frac

        // Modulated LR
        bc1 = 1 - beta1^step
        bc2 = 1 - beta2^step
        lr_base = lr * sqrt(bc2) / bc1
        glr = lr_base * exp(-H / T_epoch)

        // Update (per-element, with per-group glr)
        p = p - glr * m / (sqrt(v) + eps)
```

Propiedades:
- Cuando T_epoch -> inf: exp(-H/T) -> 1 para todo H, reduce a Adam estandar
- Cuando T_epoch -> 0: solo grupos con H~0 (signos 100% consistentes) se actualizan
- gs=64 produce ~54K grupos independientes para 3.5M params

---

## 7. Resumen

### Lo que Flux NO es:
- No es un Transformer (no usa attention)
- No es un SSM clasico (no discretiza un sistema continuo)
- No es una variante de Mamba (parametrizacion distinta del decay)
- No es FNet (WHT en features, no en tokens; es causal)

### Lo que Flux ES:
- Un modelo recurrente lineal con feature mixing via WHT
- Tres escalas temporales (fast + slow + SPM) con gating dinamico
- Content-dependent decay con modulacion element-wise (no proyecciones)
- Inference O(1) por token (estado recurrente constante)
- Parametricamente eficiente: ~271K params/capa vs ~2M en Transformer equivalente

### Innovaciones verificables:
1. WHT como unico mecanismo de feature mixing en un LM (sin MLP, sin attention)
2. Spectral gating aprendible: tanh(s * WHT(x) + b)
3. Modulacion multiplicativa del decay (O(d) vs O(d^2) de Mamba)
4. Arquitectura tri-escala dentro de una capa (fast + slow + SPM)
5. EntropicAdam: LR modulado por entropia de signos del gradiente

### Limitaciones conocidas:
1. WHT fija — no puede aprender transformadas arbitrarias
2. Sin cross-feature interactions en el decay (element-wise, no matrix)
3. d <= 1024 en kernels CUDA (max threads per block)
4. Parallel scan O(T log T) work vs O(T) del scan secuencial
5. Capacidad limitada: 3.5M/7M params vs billones en LLMs modernos
