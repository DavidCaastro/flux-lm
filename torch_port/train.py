#!/usr/bin/env python3
"""Flux v3 training with early stopping support.

Watchdog thread kills the process if no heartbeat for 180s.
"""

import multiprocessing
import os
import signal
import sys
import math
import time
import argparse
import threading
import traceback
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from flux.model import FluxModel, reset_debug_counters
from flux.optim import EntropicAdam, WarmRestartCosineSchedule
from flux.data import ByteCorpusDataset, load_corpus
from flux.checkpoint import save_pytorch, load_pytorch, load_rust_checkpoint

LN2 = 0.6931471805599453

# Global flag for graceful shutdown
_shutdown_requested = False
_main_pid = None

# ── Watchdog ──────────────────────────────────────────────────────────
_last_heartbeat = time.time()
_heartbeat_label = "init"
WATCHDOG_TIMEOUT = 180  # seconds without heartbeat → kill


def heartbeat(label):
    global _last_heartbeat, _heartbeat_label
    _last_heartbeat = time.time()
    _heartbeat_label = label


def tlog(msg):
    """Timestamped log with milliseconds, always flushed."""
    t = time.time()
    ts = time.strftime('%H:%M:%S', time.localtime(t))
    ms = int((t % 1) * 1000)
    print(f'[{ts}.{ms:03d}] {msg}', flush=True)


def _watchdog_fn():
    """Background thread: kills process if no heartbeat for WATCHDOG_TIMEOUT seconds."""
    while True:
        time.sleep(5)
        elapsed = time.time() - _last_heartbeat
        if elapsed > WATCHDOG_TIMEOUT:
            tlog("=" * 70)
            tlog(f"WATCHDOG TIMEOUT: {elapsed:.0f}s sin heartbeat!")
            tlog(f"Ultimo heartbeat: '{_heartbeat_label}'")
            tlog("Stack traces de todos los threads:")
            for tid, frame in sys._current_frames().items():
                tlog(f"  --- Thread {tid} ---")
                for line in traceback.format_stack(frame):
                    for subline in line.strip().split('\n'):
                        tlog(f"    {subline}")
            tlog("=" * 70)
            tlog("MATANDO PROCESO POR TIMEOUT")
            os._exit(1)


def vram():
    """Current VRAM usage string."""
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
    return "N/A"


def _signal_handler(signum, frame):
    global _shutdown_requested
    if os.getpid() == _main_pid:
        _shutdown_requested = True


def _worker_init_fn(worker_id):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


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
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--early-stop-bpb', type=float, default=None,
                   help='Stop training when test_bpb drops below this value')
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
        tlog(msg)


def preflight(model, device, batch_size, seq_len, dtype, rank):
    """Quick forward+backward check before training starts."""
    log('Preflight check ...', rank)
    heartbeat("preflight: creating random tensors")

    bs = min(batch_size, 4)
    tlog(f"  preflight: creando tensores x,y shape=({bs}, {seq_len}) en {device}")
    x = torch.randint(0, 256, (bs, seq_len), device=device)
    y = torch.randint(0, 256, (bs, seq_len), device=device)
    tlog(f"  preflight: tensores creados, VRAM={vram()}")

    ptdtype = {'fp32': torch.float32, 'bf16': torch.bfloat16,
               'fp16': torch.float16}[dtype]
    use_amp = dtype != 'fp32' and device != 'cpu'
    ctx = torch.autocast('cuda', dtype=ptdtype) if use_amp else nullcontext()

    try:
        heartbeat("preflight: forward")
        tlog(f"  preflight: FORWARD (AMP={use_amp}, dtype={ptdtype})...")
        t0 = time.time()
        with ctx:
            _, loss = model(x, y)
        torch.cuda.synchronize()
        tlog(f"  preflight: FORWARD OK en {time.time()-t0:.3f}s, "
             f"loss={loss.item():.4f}, VRAM={vram()}")

        heartbeat("preflight: backward")
        tlog(f"  preflight: BACKWARD...")
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        tlog(f"  preflight: BACKWARD OK en {time.time()-t0:.3f}s, VRAM={vram()}")

        heartbeat("preflight: zero_grad")
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1024**3
        tlog(f"  preflight: PASS — loss={loss.item():.4f}, peak_vram={mem:.1f}GB")
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        tlog(f"  preflight: FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    del x, y
    torch.cuda.empty_cache()
    heartbeat("preflight: done")


def main():
    global _shutdown_requested, _main_pid
    _main_pid = os.getpid()

    # Start watchdog
    wd_thread = threading.Thread(target=_watchdog_fn, daemon=True)
    wd_thread.start()
    tlog(f"Watchdog thread iniciado (timeout={WATCHDOG_TIMEOUT}s)")
    heartbeat("main: parsing args")

    args = parse_args()
    tlog(f"Args: {vars(args)}")

    heartbeat("main: setup DDP")
    rank, local_rank, world_size, device, ddp = setup_ddp()
    is_master = rank == 0
    tlog(f"DDP: rank={rank}, local_rank={local_rank}, world={world_size}, "
         f"device={device}, ddp={ddp}")

    heartbeat("main: seeds + CUDA info")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        tlog(f"CUDA: {torch.cuda.get_device_name()}, "
             f"CC={torch.cuda.get_device_capability()}, VRAM={vram()}")

    # ── Precision ──
    ptdtype = {'fp32': torch.float32, 'bf16': torch.bfloat16,
               'fp16': torch.float16}[args.dtype]
    use_amp = args.dtype != 'fp32' and device != 'cpu'
    amp_ctx = (torch.autocast(device_type='cuda', dtype=ptdtype)
               if use_amp else nullcontext())
    scaler = torch.GradScaler('cuda', enabled=(args.dtype == 'fp16'))
    tlog(f"Precision: dtype={args.dtype}, ptdtype={ptdtype}, "
         f"use_amp={use_amp}, scaler_enabled={args.dtype == 'fp16'}")

    # ── Data ──
    heartbeat("main: loading corpus")
    tlog(f"Cargando corpus desde {args.corpus}...")
    t0 = time.time()
    train_data, test_data = load_corpus(args.corpus)
    tlog(f"Corpus cargado en {time.time()-t0:.2f}s: "
         f"train={len(train_data)} bytes, test={len(test_data)} bytes")

    heartbeat("main: creating datasets")
    tlog("Creando datasets...")
    train_ds = ByteCorpusDataset(train_data, args.seq_len)
    test_ds = ByteCorpusDataset(test_data, args.seq_len)
    tlog(f"Datasets: train={len(train_ds)} chunks, test={len(test_ds)} chunks")

    heartbeat("main: creating dataloaders")
    tlog(f"Creando DataLoaders (num_workers={args.num_workers})...")
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    wk = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=wk, pin_memory=True, drop_last=True,
        persistent_workers=wk > 0,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(wk, 2), pin_memory=True, drop_last=True,
        persistent_workers=min(wk, 2) > 0,
        worker_init_fn=_worker_init_fn,
    )
    tlog(f"DataLoaders OK: train={len(train_loader)} batches, "
         f"test={len(test_loader)} batches")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── Model ──
    heartbeat("main: creating model")
    tlog("Creando modelo...")
    start_epoch = 0
    if args.resume and args.ckpt and os.path.exists(args.ckpt):
        model, ckpt_info = load_pytorch(args.ckpt, device='cpu')
        start_epoch = ckpt_info['epoch']
        tlog(f'Resumed from {args.ckpt} (epoch {start_epoch})')
    elif args.rust_ckpt:
        model, ckpt_info = load_rust_checkpoint(args.rust_ckpt)
        start_epoch = ckpt_info['epoch']
        tlog(f'Loaded Rust ckpt {args.rust_ckpt} (epoch {start_epoch})')
    else:
        model = FluxModel(d=args.d, n_layers=args.layers,
                          parallel=args.parallel,
                          n_corrections=args.n_corrections)
    tlog("Modelo creado en CPU")

    heartbeat("main: model to device")
    tlog(f"Moviendo modelo a {device}...")
    t0 = time.time()
    model = model.to(device)
    tlog(f"Modelo en {device} en {time.time()-t0:.2f}s, VRAM={vram()}")

    n_params = model.count_params()
    mode = 'parallel' if args.parallel else 'sequential'
    tlog(f'Flux v3 [{args.dtype}]: d={model.d}, L={model.n_layers}, '
         f'params={n_params:,}, mode={mode}')

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
        tlog('Gradient checkpointing enabled')

    # Preflight
    preflight(model, device, args.batch_size, args.seq_len, args.dtype, rank)

    # Reset debug counters so training logging starts fresh
    reset_debug_counters()

    if args.compile and hasattr(torch, 'compile'):
        heartbeat("main: torch.compile")
        tlog("torch.compile...")
        t0 = time.time()
        model = torch.compile(model)
        tlog(f'torch.compile OK en {time.time()-t0:.2f}s')

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    # ── Optimizer ──
    heartbeat("main: creating optimizer")
    tlog("Creando EntropicAdam + WarmRestartCosineSchedule...")
    optimizer = EntropicAdam(
        raw_model.parameters(), lr=args.lr, total_epochs=args.epochs)
    schedule = WarmRestartCosineSchedule(
        total_epochs=args.epochs, start_epoch=start_epoch)
    tlog("Optimizer y schedule creados")

    # ── WandB ──
    if args.wandb and is_master:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(raw_model, log_freq=100)

    # ── Training loop ──
    os.makedirs(args.ckpt_dir, exist_ok=True)
    tlog(f'Device: {device}, DDP: {ddp}, World: {world_size}, AMP: {use_amp}')
    tlog("=" * 70)
    tlog("INICIANDO TRAINING LOOP")
    tlog("=" * 70)
    t_start = time.time()
    best_test_bpb = float('inf')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if _shutdown_requested:
            tlog('Shutdown signal, saving checkpoint...')
            if is_master:
                ckpt_path = os.path.join(
                    args.ckpt_dir, f'flux_interrupt_{epoch-1:04d}.pt')
                save_pytorch(raw_model, optimizer, epoch - 1,
                             best_test_bpb, ckpt_path)
                tlog(f'  saved: {ckpt_path}')
            break

        t_epoch = time.time()
        heartbeat(f"epoch {epoch}: start")
        tlog(f"{'='*50} EPOCH {epoch}/{args.epochs} {'='*50}")

        if ddp:
            train_sampler.set_epoch(epoch)

        lr_scale = schedule.get_factor(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr * lr_scale
        tlog(f"  lr_scale={lr_scale:.6f}, lr_eff={args.lr * lr_scale:.6e}")

        # ── Train ──
        heartbeat(f"epoch {epoch}: train start")
        tlog(f"  TRAIN: entrando train_one_epoch...")
        model.train()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, amp_ctx,
            device, args.grad_accum, args.max_grad_norm,
            args.weight_decay, epoch,
        )
        heartbeat(f"epoch {epoch}: train done")
        tlog(f"  TRAIN OK: loss={train_loss:.6f}, VRAM={vram()}")

        # ── Eval ──
        test_loss = None
        if len(test_ds) > 0:
            heartbeat(f"epoch {epoch}: eval start")
            tlog(f"  EVAL: entrando eval_loss...")
            model.eval()
            test_loss = eval_loss(model, test_loader, amp_ctx, device, epoch)
            heartbeat(f"epoch {epoch}: eval done")
            tlog(f"  EVAL OK: loss={test_loss:.6f}")

        epoch_s = time.time() - t_epoch
        elapsed = time.time() - t_start
        h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)

        train_bpb = train_loss / LN2
        test_bpb = test_loss / LN2 if test_loss is not None else None
        if test_bpb is not None and test_bpb < best_test_bpb:
            best_test_bpb = test_bpb

        if is_master and (epoch % args.print_every == 0
                          or epoch == start_epoch + 1):
            tokens_per_sec = len(train_ds) * args.seq_len / epoch_s
            msg = f'epoch {epoch:4d}  train_bpb={train_bpb:.3f}'
            if test_bpb is not None:
                msg += f'  test_bpb={test_bpb:.3f}'
            msg += (f'  lr={lr_scale:.4f}  {epoch_s:.1f}s  '
                    f'{tokens_per_sec/1000:.0f}k tok/s  [{h:02d}:{m:02d}:{s:02d}]')
            tlog(msg)

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

        # ── Checkpoint ──
        if is_master:
            if (args.ckpt_every > 0 and epoch % args.ckpt_every == 0) \
                    or epoch == args.epochs:
                heartbeat(f"epoch {epoch}: saving ckpt")
                ckpt_path = os.path.join(
                    args.ckpt_dir, f'flux_epoch_{epoch:04d}.pt')
                tlog(f"  Guardando checkpoint: {ckpt_path}...")
                save_pytorch(raw_model, optimizer, epoch, train_loss, ckpt_path)
                tlog(f'  ckpt guardado OK')

        heartbeat(f"epoch {epoch}: done")
        tlog(f"  EPOCH {epoch} COMPLETADA en {epoch_s:.1f}s, VRAM={vram()}")

        # ── Early stopping ──
        if args.early_stop_bpb is not None and test_bpb is not None:
            if test_bpb < args.early_stop_bpb:
                tlog(f"  EARLY STOP: test_bpb={test_bpb:.4f} < {args.early_stop_bpb}")
                if is_master:
                    ckpt_path = os.path.join(
                        args.ckpt_dir, f'flux_early_stop_epoch_{epoch:04d}.pt')
                    save_pytorch(raw_model, optimizer, epoch, train_loss, ckpt_path)
                    tlog(f'  Checkpoint early-stop guardado: {ckpt_path}')
                break

    # ── Cleanup ──
    if ddp:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    tlog('Training complete.')


def train_one_epoch(model, loader, optimizer, scaler, amp_ctx,
                    device, grad_accum, max_grad_norm, wd, epoch):
    running_loss = torch.tensor(0.0, device=device)
    n_batches = 0
    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    tlog(f"    train_one_epoch: {total_batches} batches, "
         f"grad_accum={grad_accum}")

    t_data_start = time.time()

    for step, (x, y) in enumerate(loader):
        t_data = time.time() - t_data_start
        t_step_start = time.time()
        heartbeat(f"epoch {epoch}, batch {step+1}/{total_batches}")

        # Verbose: first batch, every 2000th, and last batch
        verbose = (step == 0) or (step % 2000 == 0) or (step == total_batches - 1)

        bp = f"E{epoch} B{step+1}/{total_batches}"

        if verbose:
            tlog(f"    [{bp}] x.shape={list(x.shape)}, "
                 f"data_load={t_data:.3f}s, VRAM={vram()}")

        # ── To device ──
        if verbose:
            tlog(f"    [{bp}] x,y → {device}...")
        t0 = time.time()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        t_transfer = time.time() - t0
        if verbose:
            tlog(f"    [{bp}] transfer={t_transfer:.4f}s")

        # ── Forward ──
        if verbose:
            tlog(f"    [{bp}] FORWARD start...")
        t0 = time.time()
        try:
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
        except Exception as e:
            tlog(f"    [{bp}] FORWARD EXCEPTION: {e}")
            traceback.print_exc()
            raise

        if verbose:
            torch.cuda.synchronize()
            t_fwd = time.time() - t0
            loss_val = loss.item() * grad_accum
            tlog(f"    [{bp}] FORWARD done={t_fwd:.3f}s, "
                 f"loss={loss_val:.4f}, VRAM={vram()}")

        # ── Backward ──
        if verbose:
            tlog(f"    [{bp}] BACKWARD start...")
        t0 = time.time()
        try:
            scaler.scale(loss).backward()
        except Exception as e:
            tlog(f"    [{bp}] BACKWARD EXCEPTION: {e}")
            traceback.print_exc()
            raise

        if verbose:
            torch.cuda.synchronize()
            t_bwd = time.time() - t0
            tlog(f"    [{bp}] BACKWARD done={t_bwd:.3f}s, VRAM={vram()}")

        # ── Optimizer step ──
        if (step + 1) % grad_accum == 0 or (step + 1) == total_batches:
            if verbose:
                tlog(f"    [{bp}] OPTIMIZER step start...")
            t0 = time.time()
            try:
                scaler.unscale_(optimizer)
                raw = model.module if hasattr(model, 'module') else model
                gn = torch.nn.utils.clip_grad_norm_(raw.parameters(),
                                                     max_grad_norm)
                if verbose:
                    tlog(f"    [{bp}] grad_norm={gn:.4f}, "
                         f"calling scaler.step...")
                scaler.step(optimizer, epoch=epoch)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if wd > 0:
                    with torch.no_grad():
                        for p in raw.parameters():
                            p.mul_(1.0 - wd)
            except Exception as e:
                tlog(f"    [{bp}] OPTIMIZER EXCEPTION: {e}")
                traceback.print_exc()
                raise

            if verbose:
                torch.cuda.synchronize()
                t_opt = time.time() - t0
                tlog(f"    [{bp}] OPTIMIZER done={t_opt:.3f}s, VRAM={vram()}")

                # NaN check only on last batch of epoch
                if step == total_batches - 1:
                    has_nan = any(p.isnan().any().item() for p in raw.parameters())
                    if has_nan:
                        tlog(f"    [{bp}] *** NaN DETECTED in weights!")

        # Accumulate loss
        running_loss += loss.detach() * grad_accum
        n_batches += 1

        if verbose:
            t_total = time.time() - t_step_start
            tlog(f"    [{bp}] TOTAL step={t_total:.3f}s")

        t_data_start = time.time()

    avg_loss = running_loss.item() / max(n_batches, 1)
    tlog(f"    train_one_epoch: {n_batches} batches, avg_loss={avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def eval_loss(model, loader, amp_ctx, device, epoch=0):
    total_loss = 0.0
    n_batches = 0
    total = len(loader)

    for step, (x, y) in enumerate(loader):
        heartbeat(f"epoch {epoch}, eval batch {step+1}/{total}")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        try:
            with amp_ctx:
                _, loss = model(x, y)
        except Exception as e:
            tlog(f"    eval batch {step+1}: EXCEPTION: {e}")
            traceback.print_exc()
            raise

        total_loss += loss.item()
        n_batches += 1

        if step == 0 or step == total - 1:
            tlog(f"    eval {step+1}/{total}: loss={loss.item():.4f}")

    avg = total_loss / max(n_batches, 1)
    tlog(f"    eval_loss: {n_batches} batches, avg={avg:.6f}")
    return avg


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
