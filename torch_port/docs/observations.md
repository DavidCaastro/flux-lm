# Flux v3 — Observaciones y Lecciones Aprendidas

## Bugs y correcciones

### Bug 1: `_popcount32` overflow int64 → NaN (commit 2e82698)
- **Síntoma**: NaN en forward pass
- **Causa**: Aritmética entera en `_popcount32` desbordaba int32
- **Fix**: Uso de int64 / corrección de la aritmética
- **Lección**: Siempre verificar rangos de tipos enteros en operaciones bitwise

### Bug 2: `p_ratio` clamp insuficiente → NaN (commit 2e82698)
- **Síntoma**: NaN propagado desde EntropicAdam
- **Causa**: `clamp(1e-10)` se redondea a 1.0 en float32 → `log(0) = -inf → NaN`
- **Fix**: Clamp con valor más grande que la precisión de float32
- **Lección**: Los clamps near-zero en float32 son peligrosos — usar valores > 1e-7

### Bug 3: `grad_decay = None` en backward (commit 995e7f1)
- **Síntoma**: `delta_fast` y `delta_slow` no aprendían (gradientes = None)
- **Causa**: `_ParallelScanFunction.backward` no computaba grad_decay
- **Fix**: Backward analítico para grad_decay, nunca retorna None
- **Lección**: SIEMPRE verificar que `param.grad is not None` para todos los parámetros después de backward. Test explícito obligatorio.

### Bug 4: Weight decay sin escalar por lr_scale (Run 4, 2026-09-17)
- **Síntoma**: test_loss degradó 0.668 → 0.840 en 3 epochs tras warm restart
- **Causa**: Weight decay fijo (1e-5) aplicado cuando lr→0 decaía los pesos ~3.4%/epoch sin que el optimizer pudiera compensar (lr=0 → no hay actualización de gradiente)
- **Fix**: `wd_eff = args.weight_decay * lr_scale`
- **Lección**: En schedules con lr→0 (coseno, SGDR), el weight decay DEBE ser proporcional al LR. Decoupled weight decay ≠ weight decay independiente.

### Bug 5: sign_history int64 corrupto por load_state_dict (Run 4, 2026-09-17)
- **Síntoma**: Comportamiento errático del optimizer post-resume
- **Causa**: `optimizer.load_state_dict()` convierte int64 → float32. Valores enteros > 2²⁴ (16,777,216) pierden precisión en float32
- **Fix**: Backup de tensores int64 antes de load_state_dict, restore después
- **Lección**: PyTorch load_state_dict no preserva dtypes especiales. Cualquier estado del optimizer que no sea float32 requiere manejo manual.

---

## Observaciones de entrenamiento

### Estancamiento con decay constante (epochs 25-30)
- test_bpb oscilando 1.045-1.054, mejora Δ≈0.003/epoch
- **Causa raíz**: Decay fijo impone compromiso irreconciliable
  - Fast filters (λ~0.1): capturan patrones locales, olvidan contexto
  - Slow filters (λ~0.99): retienen contexto, no reaccionan a cambios
  - No hay punto intermedio óptimo — depende del contenido
- **Solución**: Selective scan (decay content-dependent) → rompió estancamiento

### Selective scan: mejora inmediata (epochs 31+)
- Init δ_mod = zeros → comportamiento inicial = idéntico a pre-cambio
- Mejora 0.016 BPB/epoch vs 0.003/epoch (5x más rápido)
- El modelo aprende rápidamente a modular el decay según contenido
- **Validación**: La hipótesis del decay adaptativo era correcta

### SGDR no funciona post-convergencia con LR alto (epochs 51-77)
- LR pico 3e-4 en modelo de 3.5M params convergido → destructivo
- Warm restart empuja al modelo fuera del valley, pero no puede regresar
- Cada ciclo empeora: ciclo 1 fondo=0.668, ciclo 2 fondo=0.733, ciclo 3→0.742
- grad_norm se dispara (43 vs 6-10 normal)
- **Conclusión**: SGDR requiere LR pico decreciente por ciclo, o modelo con capacidad para absorber la perturbación
- **Alternativa**: SWA (promediar checkpoints) captura ~80% del beneficio sin riesgo

### Velocidad por implementación del scan

| Implementación | Tiempo/batch | Tokens/s | Speedup |
|----------------|-------------|----------|---------|
| PyTorch puro (F.pad + loops) | 0.81s | ~20k | 1x |
| Kernel CUDA fusionado | 0.52s | ~55k | 1.56x total |
| Solo kernel vs scan PyTorch | 0.45ms vs 3.24ms | — | **7.27x** |

### Uso de VRAM (B=64, T=256, d=512, bf16)

| Fase | VRAM |
|------|------|
| Forward peak | 8.82 GB |
| Backward peak | ~10 GB |
| Post-step (idle) | 0.06 GB |
| Con grad-checkpoint | 1.28 GB peak |
| Disponible (RTX 4090) | 23.5 GB |

Margen amplio: ~13 GB libres. Permite escalar a d=1024 (~40 GB estimado con grad-checkpoint).

---

## Observaciones de infraestructura

### Vast.ai + NGC container
- Imagen NGC nv26.08 tiene TODO para JIT: nvcc 13.4, gcc 13.3, ninja 1.13
- sm_89 NO en PyTorch arch list built-in → usa sm_86 fallback
- Nuestros kernels compilan para sm_89 nativo → mejor rendimiento que PyTorch built-in
- Disco 32 GB overlay: gestionable con ~100 MB corpus + ~500 MB checkpoints

### Notas operativas críticas
- **SIEMPRE** usar `--parallel` en train.py (sin él: modo secuencial ~20x más lento)
- **SIEMPRE** limpiar `flux/__pycache__/` al actualizar código Python en vast
- **SIEMPRE** limpiar `.selective_scan_cache/` y `.kernel_cache/` si se modifica código CUDA
- Verificar `kill -9 <PID>` + `nvidia-smi` para confirmar GPU libre antes de nuevo run
- Log verbose solo en batch 0, batch%2000, último batch

### Checkpoints
- ~42 MB por checkpoint (d=512, L=12)
- Incluyen: model_state_dict, optimizer_state_dict, epoch, best_loss
- Compatible con strict=False para nuevos parámetros (δ_mod)
- sign_history (int64) requiere backup/restore manual en resume

---

## Límites teóricos

### Shannon entropy floor
- **Teórico (modelo infinito)**: ~0.3-0.5 BPB para código Python
  - Python tiene alta redundancia: indentación, keywords, naming conventions
  - Entropía condicional del inglés: ~0.7-1.0 bits/char
  - Código tiene más estructura → menor entropía
- **Práctico (3.5M params)**: ~0.6-0.7 BPP
  - Limitado por capacidad del modelo
  - Más params → más cercano al límite teórico
- **Estado actual**: 0.963 BPB
- **Gap aprovechable**: ~0.26 BPB con más datos/params

### Scaling laws (estimaciones)
- d=512→1024 (~14M params, 4x): ~0.15-0.25 BPP mejora → ~0.7-0.8 BPB
- Corpus 60MB→1GB (17x): ~0.05-0.10 BPB mejora
- Ambos combinados: ~0.65-0.75 BPB estimado
