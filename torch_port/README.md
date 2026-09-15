# Flux v3 — PyTorch GPU Port

Port completo del modelo Flux v3 byte-level language model de Rust a PyTorch,
optimizado para entrenamiento en GPU cloud.

## Arquitectura preservada

- **Walsh-Hadamard Transform** (WHT): mezcla sin parametros O(d log d)
- **Content-dependent gating**: gate por byte + estado recurrente
- **Dual-timescale memory**: filtros exponenciales fast/slow con decay learnable via softplus
- **Semantic Partition Module** (SPM): K=4 features semanticos con stride=4
- **Residual scaling**: 1/ln(L+2) por capa
- **EntropicAdam**: Adam con LR per-group basado en entropia de signo del gradiente

## Capacidades GPU

| Feature | Soporte |
|---------|---------|
| Single GPU | `python train.py` |
| Multi-GPU DDP | `torchrun --nproc_per_node=N train.py` |
| Mixed precision (bf16/fp16) | `--dtype bf16` |
| Gradient checkpointing | `--grad-checkpoint` |
| Gradient accumulation | `--grad-accum N` |
| torch.compile | `--compile` |
| WandB logging | `--wandb` |
| Checkpoint save/resume | `--ckpt PATH --resume` |
| Rust checkpoint import | `--rust-ckpt PATH` |
| Docker + CUDA | `Dockerfile` |

## Inicio rapido

```bash
pip install -r requirements.txt

# Entrenar en GPU
python train.py --corpus ../bench/corpus_multilingual.txt \
    --d 256 --layers 6 --epochs 200 --batch-size 64 \
    --dtype bf16 --compile --wandb

# Multi-GPU
torchrun --nproc_per_node=4 train.py \
    --corpus data/large_corpus.txt \
    --d 512 --layers 12 --epochs 500 \
    --batch-size 128 --grad-accum 4 \
    --dtype bf16 --compile --grad-checkpoint --wandb

# Generar texto
python generate_cli.py --ckpt checkpoints/flux_epoch_0200.pt \
    --seed-text "Once upon" --length 1000 --temperature 0.7
```

## Docker

```bash
docker build -t flux-lm .

# Single GPU
docker run --gpus all -v $(pwd)/../bench:/data flux-lm \
    train.py --corpus /data/corpus_multilingual.txt --d 256 --layers 6

# Multi-GPU
docker run --gpus all -v /data:/data flux-lm \
    -m torch.distributed.run --nproc_per_node=4 \
    train.py --corpus /data/corpus.txt --d 512 --layers 12
```

## Compatibilidad con checkpoints Rust

Puedes cargar checkpoints binarios V3 del modelo Rust:

```bash
# Importar checkpoint Rust y continuar en GPU
python train.py --corpus data.txt --rust-ckpt ../model.bin --epochs 500

# Generar desde checkpoint Rust
python generate_cli.py --rust-ckpt ../model.bin --seed-text "Hello" --length 500
```

## Estructura

```
torch_port/
├── flux/
│   ├── model.py        # FluxModel, FluxLayer, WHT
│   ├── optim.py        # EntropicAdam + WarmRestartCosineSchedule
│   ├── data.py         # ByteCorpusDataset
│   ├── generate.py     # Autoregressive generation
│   └── checkpoint.py   # Save/load + Rust format interop
├── train.py            # Training script (DDP, AMP, wandb)
├── generate_cli.py     # Generation CLI
├── Dockerfile          # GPU cloud container
└── requirements.txt
```
