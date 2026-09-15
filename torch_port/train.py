#!/usr/bin/env python3
"""Flux v3 training — GPU-ready with DDP, AMP, gradient accumulation, wandb.

Usage:
  Single GPU:
    python train.py --corpus data.txt --d 256 --layers 6

  Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train.py --corpus data.txt --d 512 --layers 12

  Resume from checkpoint:
    python train.py --corpus data.txt --ckpt flux_model.pt --resume
"""

import os
import signal
import sys
import math
import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from flux.model import FluxModel
from flux.optim import EntropicAdam, WarmRestartCosineSchedule
from flux.data import ByteCorpusDataset, load_corpus
from flux.checkpoint import save_pytorch, load_pytorch, load_rust_checkpoint

LN2 = 0.6931471805599453

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def parse_args():
    p = argparse.ArgumentParser(description='Flux v3 Training')
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--corpus', type=str, required=True)
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--max-grad-norm', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--grad-checkpoint', action='store_true')
    p.add_argument('--parallel', action='store_true')
    p.add_argument('--n-corrections', type=int, default=1)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--rust-ckpt', type=str, default=None)
    p.add_argument('--ckpt-every', type=int, default=10)
    p.add_argument('--ckpt-dir', type=str, default='checkpoints')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='flux-lm')
    p.add_argument('--print-every', type=int, default=5)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


def setup_ddp():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        torch.distributed.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return rank, local_rank, world_size, device, ddp


def log(msg, rank=0):
    if rank == 0:
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{ts}] {msg}', flush=True)


def preflight(model, device, batch_size, seq_len, dtype, rank):
    """Quick forward+backward check before training starts."""
    log('Preflight check ...', rank)
    x = torch.randint(0, 256, (min(batch_size, 4), seq_len), device=device)
    y = torch.randint(0, 256, (min(batch_size, 4), seq_len), device=device)
    ptdtype = {'fp32': torch.float32, 'bf16': torch.bfloat16,
               'fp16': torch.float16}[dtype]
    use_amp = dtype != 'fp32' and device != 'cpu'
    ctx = torch.autocast('cuda', dtype=ptdtype) if use_amp else nullcontext()
    try:
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        log(f'  OK: loss={loss.item():.4f}, peak_vram={mem:.1f}GB', rank)
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError as e:
        log(f'  FAILED: {e}', rank)
        sys.exit(1)
    del x, y
    torch.cuda.empty_cache()


def main():
    global _shutdown_requested
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    rank, local_rank, world_size, device, ddp = setup_ddp()
    is_master = rank == 0

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Precision ──
    ptdtype = {'fp32': torch.float32, 'bf16': torch.bfloat16,
               'fp16': torch.float16}[args.dtype]
    use_amp = args.dtype != 'fp32' and device != 'cpu'
    amp_ctx = (torch.autocast(device_type='cuda', dtype=ptdtype)
               if use_amp else nullcontext())
    scaler = torch.GradScaler('cuda', enabled=(args.dtype == 'fp16'))

    # ── Data ──
    train_data, test_data = load_corpus(args.corpus)
    train_ds = ByteCorpusDataset(train_data, args.seq_len)
    test_ds = ByteCorpusDataset(test_data, args.seq_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    wk = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=wk, pin_memory=True, drop_last=True,
        persistent_workers=wk > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(wk, 2), pin_memory=True, drop_last=True,
        persistent_workers=min(wk, 2) > 0,
    )

    log(f'Corpus: {len(train_data)+len(test_data)} bytes '
        f'(train={len(train_data)}, test={len(test_data)}), '
        f'seq_len={args.seq_len}', rank)
    log(f'Train: {len(train_ds)} chunks, {len(train_ds)//args.batch_size} '
        f'batches/epoch | Test: {len(test_ds)} chunks', rank)

    # ── Model ──
    start_epoch = 0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        model, ckpt_info = load_pytorch(args.ckpt, device='cpu')
        start_epoch = ckpt_info['epoch']
        log(f'Resumed from {args.ckpt} (epoch {start_epoch})', rank)
    elif args.rust_ckpt:
        model, ckpt_info = load_rust_checkpoint(args.rust_ckpt)
        start_epoch = ckpt_info['epoch']
        log(f'Loaded Rust ckpt {args.rust_ckpt} (epoch {start_epoch})', rank)
    else:
        model = FluxModel(d=args.d, n_layers=args.layers,
                          parallel=args.parallel,
                          n_corrections=args.n_corrections)

    model = model.to(device)
    n_params = model.count_params()
    mode = 'parallel' if args.parallel else 'sequential'
    log(f'Flux v3 [{args.dtype}]: d={model.d}, L={model.n_layers}, '
        f'params={n_params:,}, mode={mode}', rank)

    if args.parallel and (args.resume or args.rust_ckpt):
        model.set_mode(parallel=True, n_corrections=args.n_corrections)

    if args.grad_checkpoint:
        for layer in model.layers:
            layer._orig_forward = layer.forward
            def make_ckpt_fwd(mod):
                def ckpt_fwd(*a, **kw):
                    return torch.utils.checkpoint.checkpoint(
                        mod._orig_forward, *a, use_reentrant=False, **kw)
                return ckpt_fwd
            layer.forward = make_ckpt_fwd(layer)
        log('Gradient checkpointing enabled', rank)

    # Preflight: quick sanity check before committing to training
    preflight(model, device, args.batch_size, args.seq_len, args.dtype, rank)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        log('torch.compile enabled', rank)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    # ── Optimizer ──
    optimizer = EntropicAdam(
        raw_model.parameters(), lr=args.lr, total_epochs=args.epochs)
    schedule = WarmRestartCosineSchedule(
        total_epochs=args.epochs, start_epoch=start_epoch)

    # ── WandB ──
    if args.wandb and is_master:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(raw_model, log_freq=100)

    # ── Training loop ──
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log(f'Device: {device}, DDP: {ddp}, World: {world_size}, AMP: {use_amp}',
        rank)
    t_start = time.time()
    best_test_bpb = float('inf')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if _shutdown_requested:
            log('Shutdown signal received, saving checkpoint ...', rank)
            if is_master:
                ckpt_path = os.path.join(
                    args.ckpt_dir, f'flux_interrupt_{epoch-1:04d}.pt')
                save_pytorch(raw_model, optimizer, epoch - 1,
                             best_test_bpb, ckpt_path)
                log(f'  saved: {ckpt_path}', rank)
            break

        t_epoch = time.time()

        if ddp:
            train_sampler.set_epoch(epoch)

        lr_scale = schedule.get_factor(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_scale

        # Train
        model.train()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, amp_ctx,
            device, args.grad_accum, args.max_grad_norm,
            args.weight_decay, epoch,
        )

        # Eval
        test_loss = None
        if len(test_ds) > 0:
            model.eval()
            test_loss = eval_loss(model, test_loader, amp_ctx, device)

        epoch_s = time.time() - t_epoch
        elapsed = time.time() - t_start
        h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)

        train_bpb = train_loss / LN2
        test_bpb = test_loss / LN2 if test_loss is not None else None
        if test_bpb is not None and test_bpb < best_test_bpb:
            best_test_bpb = test_bpb

        if is_master and (epoch % args.print_every == 0 or epoch == start_epoch + 1):
            tokens_per_sec = len(train_ds) * args.seq_len / epoch_s
            msg = (f'epoch {epoch:4d}  train_bpb={train_bpb:.3f}')
            if test_bpb is not None:
                msg += f'  test_bpb={test_bpb:.3f}'
            msg += (f'  lr={lr_scale:.4f}  {epoch_s:.1f}s  '
                    f'{tokens_per_sec/1000:.0f}k tok/s  [{h:02d}:{m:02d}:{s:02d}]')
            log(msg, rank)

            if args.wandb:
                import wandb
                log_dict = {
                    'train/loss': train_loss, 'train/bpb': train_bpb,
                    'lr_scale': lr_scale, 'epoch_s': epoch_s,
                    'tokens_per_sec': tokens_per_sec,
                }
                if test_loss is not None:
                    log_dict['test/loss'] = test_loss
                    log_dict['test/bpb'] = test_bpb
                wandb.log(log_dict, step=epoch)

        # Checkpoint
        if is_master:
            if (args.ckpt_every > 0 and epoch % args.ckpt_every == 0) \
                    or epoch == args.epochs:
                ckpt_path = os.path.join(
                    args.ckpt_dir, f'flux_epoch_{epoch:04d}.pt')
                save_pytorch(raw_model, optimizer, epoch, train_loss, ckpt_path)
                log(f'  ckpt: {ckpt_path}', rank)

    # ── Cleanup ──
    if ddp:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    log('Training complete.', rank)


def train_one_epoch(model, loader, optimizer, scaler, amp_ctx,
                    device, grad_accum, max_grad_norm, wd, epoch):
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx:
            _, loss = model(x, y)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            raw = model.module if hasattr(model, 'module') else model
            torch.nn.utils.clip_grad_norm_(raw.parameters(), max_grad_norm)

            scaler.step(optimizer, epoch=epoch)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Decoupled weight decay
            if wd > 0:
                with torch.no_grad():
                    for p in raw.parameters():
                        p.mul_(1.0 - wd)

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_loss(model, loader, amp_ctx, device):
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


if __name__ == '__main__':
    main()
