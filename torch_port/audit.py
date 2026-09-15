#!/usr/bin/env python3
"""Full pre-training audit: environment, kernels, forward/backward, optimizer, dataloader, checkpoint."""

import torch, sys, os, math, shutil

print("=" * 70)
print("AUDITORIA COMPLETA PRE-ENTRENAMIENTO")
print("=" * 70)
errors = []

# --- 1. ENTORNO GPU ---
print("\n[1] ENTORNO GPU")
assert torch.cuda.is_available(), "CUDA no disponible"
gpu = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  GPU: {gpu}")
print(f"  Compute Capability: {cap[0]}.{cap[1]}")
print(f"  VRAM: {vram:.1f} GB")
print(f"  CUDA: {torch.version.cuda}")
print(f"  PyTorch: {torch.__version__}")
print(f"  TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST', 'NO SET')}")
print(f"  bf16 soportado: {torch.cuda.is_bf16_supported()}")

if cap != (8, 9):
    errors.append(f"Compute capability {cap} != (8, 9) esperado para RTX 4090")
if not torch.cuda.is_bf16_supported():
    errors.append("bf16 no soportado")
if os.environ.get("TORCH_CUDA_ARCH_LIST") != "8.9":
    errors.append("TORCH_CUDA_ARCH_LIST no es 8.9")

# --- 2. CORPUS ---
print("\n[2] CORPUS")
corpus_path = "/workspace/corpus_100mb.txt"
if os.path.exists(corpus_path):
    size = os.path.getsize(corpus_path)
    print(f"  {corpus_path}: {size/1024/1024:.1f} MB")
    if size < 1024*1024:
        errors.append(f"Corpus demasiado pequeno: {size} bytes")
else:
    errors.append(f"Corpus no encontrado: {corpus_path}")

# --- 3. DISCO ---
print("\n[3] DISCO")
total, used, free = shutil.disk_usage("/workspace")
print(f"  Total: {total/1024**3:.1f} GB, Usado: {used/1024**3:.1f} GB, Libre: {free/1024**3:.1f} GB")
if free < 2 * 1024**3:
    errors.append(f"Disco libre insuficiente: {free/1024**3:.1f} GB")

# --- 4. COMPILACION KERNELS ---
print("\n[4] COMPILACION KERNELS CUDA")
try:
    from torch_port.flux.model import FluxModel, HAS_FUSED_KERNELS
    print(f"  Fused kernels: {HAS_FUSED_KERNELS}")
    if not HAS_FUSED_KERNELS:
        errors.append("Fused kernels no compilaron")
except Exception as e:
    errors.append(f"Error importando modelo: {e}")
    sys.exit(1)

# --- 5. CONFIGURACION EXACTA ---
print("\n[5] CONFIGURACION DE ENTRENAMIENTO")
d, layers, seq_len, batch_size = 512, 12, 256, 64
dtype_str = "bf16"
print(f"  d={d}, layers={layers}, seq_len={seq_len}, batch={batch_size}, dtype={dtype_str}")

if d > 1024:
    errors.append(f"d={d} excede limite de 1024 threads/block para WHT kernel")
if seq_len > 1024:
    errors.append(f"seq_len={seq_len} excede limite de 1024 para parallel scan kernel")
print(f"  WHT kernel: d={d} <= 1024 OK")
print(f"  Scan kernel: T={seq_len} <= 1024 OK")

# --- 6. FORWARD + BACKWARD ---
print("\n[6] FORWARD + BACKWARD (config exacta de entrenamiento)")
device = torch.device("cuda")
model = FluxModel(d=d, n_layers=layers, parallel=True, use_fused=True).to(device)
n_params = model.count_params()
print(f"  Params: {n_params:,}")

byte_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits, loss = model(byte_ids, targets)
print(f"  Forward OK: loss={loss.item():.4f}, logits shape={tuple(logits.shape)}")

loss.backward()
torch.cuda.synchronize()
peak_fwd_bwd = torch.cuda.max_memory_allocated() / 1024**3
print(f"  Backward OK: peak VRAM={peak_fwd_bwd:.2f} GB / {vram:.1f} GB")

if peak_fwd_bwd > vram * 0.95:
    errors.append(f"VRAM insuficiente: {peak_fwd_bwd:.1f}GB > 95% de {vram:.1f}GB")

# --- 7. GRADIENTES ---
print("\n[7] GRADIENTES")
total_p = 0
with_grad = 0
no_grad_names = []
for name, p in model.named_parameters():
    total_p += 1
    if p.grad is not None and torch.isfinite(p.grad).all():
        with_grad += 1
    else:
        no_grad_names.append(name)

print(f"  Con gradiente finito: {with_grad}/{total_p}")
if no_grad_names:
    errors.append(f"Params sin gradiente: {no_grad_names}")
    for n in no_grad_names:
        print(f"    SIN GRAD: {n}")

# --- 8. OPTIMIZER ---
print("\n[8] OPTIMIZER (EntropicAdam)")
from torch_port.flux.optim import EntropicAdam, WarmRestartCosineSchedule
model.zero_grad(set_to_none=True)

with torch.autocast("cuda", dtype=torch.bfloat16):
    _, loss2 = model(byte_ids, targets)
loss2.backward()

opt = EntropicAdam(model.parameters(), lr=1e-3, total_epochs=250)
try:
    opt.step(epoch=1)
    print("  step(epoch=1) OK")
except Exception as e:
    errors.append(f"Optimizer step fallo: {e}")

# --- 9. GRADIENT CHECKPOINTING ---
print("\n[9] GRADIENT CHECKPOINTING")
model.zero_grad(set_to_none=True)
for layer in model.layers:
    layer._orig_forward = layer.forward
    def make_ckpt_fwd(mod):
        def ckpt_fwd(*a, **kw):
            return torch.utils.checkpoint.checkpoint(
                mod._orig_forward, *a, use_reentrant=False, **kw)
        return ckpt_fwd
    layer.forward = make_ckpt_fwd(layer)

torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    _, loss3 = model(byte_ids, targets)
loss3.backward()
torch.cuda.synchronize()
peak_gc = torch.cuda.max_memory_allocated() / 1024**3
savings = (1 - peak_gc / peak_fwd_bwd) * 100 if peak_fwd_bwd > 0 else 0
print(f"  Peak VRAM con grad-ckpt: {peak_gc:.2f} GB (ahorro: {savings:.0f}%)")

# --- 10. DATALOADER ---
print("\n[10] DATALOADER")
from torch_port.flux.data import ByteCorpusDataset, load_corpus
train_data, test_data = load_corpus(corpus_path)
train_ds = ByteCorpusDataset(train_data, seq_len)
test_ds = ByteCorpusDataset(test_data, seq_len)
print(f"  Train chunks: {len(train_ds):,} ({len(train_ds)//batch_size} batches/epoch)")
print(f"  Test chunks:  {len(test_ds):,}")

from torch.utils.data import DataLoader
import signal
def _worker_init(wid):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True,
                    persistent_workers=True, worker_init_fn=_worker_init)
batch_x, batch_y = next(iter(loader))
print(f"  Batch shapes: x={tuple(batch_x.shape)}, y={tuple(batch_y.shape)}")
print(f"  Byte range: x=[{batch_x.min()},{batch_x.max()}], y=[{batch_y.min()},{batch_y.max()}]")
if batch_x.shape != (batch_size, seq_len):
    errors.append(f"Batch shape incorrecta: {batch_x.shape}")
del loader, batch_x, batch_y

# --- 11. CHECKPOINT SAVE/LOAD ---
print("\n[11] CHECKPOINT")
from torch_port.flux.checkpoint import save_pytorch, load_pytorch
ckpt_path = "/tmp/test_ckpt.pt"
save_pytorch(model, opt, 0, loss3.item(), ckpt_path)
ckpt_size = os.path.getsize(ckpt_path) / 1024**2
print(f"  Save OK: {ckpt_size:.1f} MB")
model2, info = load_pytorch(ckpt_path, device="cpu")
print(f"  Load OK: d={info['d']}, layers={info['n_layers']}, epoch={info['epoch']}")
os.remove(ckpt_path)
del model2

# --- 12. AMP ---
print("\n[12] AMP / GRADSCALER")
scaler = torch.GradScaler("cuda", enabled=(dtype_str == "fp16"))
print(f"  dtype={dtype_str} -> GradScaler enabled={scaler.is_enabled()}")
print(f"  bf16 no necesita GradScaler (rango dinamico suficiente) OK")

# --- 13. ESTIMACION ---
print("\n[13] ESTIMACION DE RECURSOS")
batches_per_epoch = len(train_ds) // batch_size
total_tokens = len(train_ds) * seq_len * 250
ckpts = 250 // 25
print(f"  Batches por epoch: {batches_per_epoch:,}")
print(f"  Tokens totales (250 epochs): {total_tokens/1e9:.1f}B")
print(f"  Checkpoints a guardar: {ckpts} x {ckpt_size:.0f}MB = {ckpts*ckpt_size:.0f}MB")

# --- CLEANUP ---
del model, opt, byte_ids, targets
torch.cuda.empty_cache()

# --- RESULTADO ---
print("\n" + "=" * 70)
if errors:
    print(f"FAIL: {len(errors)} problema(s) encontrado(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("PASS: READY para entrenamiento")
    print("  Comando:")
    print("  cd /workspace/flux-lm/torch_port && python3 train.py \\")
    print("    --corpus /workspace/corpus_100mb.txt \\")
    print("    --d 512 --layers 12 --epochs 250 \\")
    print("    --batch-size 64 --seq-len 256 \\")
    print("    --dtype bf16 --parallel \\")
    print("    --grad-checkpoint \\")
    print("    --ckpt-every 25 --print-every 5")
print("=" * 70)
