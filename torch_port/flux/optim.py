"""EntropicAdam — INSTRUMENTED with logging for first N steps."""

import math
import time
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

_step_count = 0
_LOG_STEPS_UNTIL = 0  # Log first N optimizer steps in detail


def _olog(msg):
    t = time.time()
    ts = time.strftime('%H:%M:%S', time.localtime(t))
    ms = int((t % 1) * 1000)
    print(f'[{ts}.{ms:03d}] [OPTIM] {msg}', flush=True)


def _popcount32(x: torch.Tensor) -> torch.Tensor:
    """Parallel bit-count for 32-bit integers stored as int64 tensors.

    The final & 0xFF is required because * 0x01010101 can overflow 32 bits
    when operating on int64, causing the >> 24 to include carry bits.
    Without this mask, popcount > 8 returns wrong values (e.g. 265 instead of 9),
    which leads to NaN in the entropy calculation.
    """
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101 >> 24) & 0xFF


class EntropicAdam(Optimizer):
    """Adam optimizer with entropic learning rate scaling.

    Fully vectorized: all sign-entropy computation runs on GPU.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 t_initial=2.0, t_final=0.5, total_epochs=50,
                 group_size=64):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        t_initial=t_initial, t_final=t_final,
                        total_epochs=max(total_epochs, 1),
                        group_size=group_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, epoch=0):
        global _step_count
        _step_count += 1
        verbose = _step_count <= _LOG_STEPS_UNTIL

        if verbose:
            _olog(f"step #{_step_count}: epoch={epoch}, "
                  f"n_param_groups={len(self.param_groups)}")
        t_total = time.time()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for gi, group in enumerate(self.param_groups):
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            gs = group['group_size']
            t_initial = group['t_initial']
            t_final = group['t_final']
            total_epochs = group['total_epochs']

            frac = epoch / total_epochs
            t_epoch = t_initial * (t_final / t_initial) ** frac

            if verbose:
                _olog(f"  group {gi}: lr={lr:.6e}, t_epoch={t_epoch:.4f}, "
                      f"n_params={sum(1 for p in group['params'] if p.grad is not None)}")

            n_params_processed = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    n_groups = (p.numel() + gs - 1) // gs
                    state['sign_history'] = torch.zeros(
                        n_groups, dtype=torch.int64, device=p.device)

                state['step'] += 1
                t = state['step']
                m, v = state['m'], state['v']

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                lr_base = lr * math.sqrt(bc2) / bc1

                # Update moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Vectorized sign-entropy
                flat_grad = grad.reshape(-1)
                n = flat_grad.numel()
                n_groups = state['sign_history'].shape[0]

                pad_size = n_groups * gs - n
                if pad_size > 0:
                    flat_padded = F.pad(flat_grad, (0, pad_size))
                else:
                    flat_padded = flat_grad

                chunks = flat_padded.reshape(n_groups, gs)
                pos_counts = (chunks > 0).sum(dim=1)
                majority = (pos_counts * 2 >= gs).long()

                sign_hist = state['sign_history']
                sign_hist.bitwise_left_shift_(1)
                sign_hist.bitwise_and_(0xFFFFFFFF)
                sign_hist.bitwise_or_(majority)

                bits = _popcount32(sign_hist).float()
                # Clamp must be wide enough for float32: 1.0-1e-10 rounds to 1.0
                # in fp32 (ULP at 1.0 ≈ 1.19e-7), causing log(0)=NaN
                p_ratio = (bits / 32.0).clamp(0.001, 0.999)
                h = -(p_ratio * p_ratio.log()
                      + (1 - p_ratio) * (1 - p_ratio).log())
                glr = lr_base * torch.exp(-h / t_epoch)

                glr_expanded = glr.repeat_interleave(gs)[:n]
                flat_p = p.reshape(-1)
                flat_m = m.reshape(-1)
                flat_v = v.reshape(-1)
                flat_p.add_(
                    glr_expanded * flat_m / (flat_v.sqrt() + eps),
                    alpha=-1.0,
                )

                n_params_processed += 1

                # Log first param of first step only
                if verbose and n_params_processed <= 2:
                    _olog(f"    param {n_params_processed}: numel={p.numel()}, "
                          f"grad_norm={grad.float().norm().item():.4f}, "
                          f"lr_base={lr_base:.6e}, "
                          f"glr_range=[{glr.min().item():.6e}, {glr.max().item():.6e}]")

            if verbose:
                _olog(f"  group {gi}: {n_params_processed} params updated")

        if verbose:
            dt = time.time() - t_total
            _olog(f"step #{_step_count} done in {dt:.4f}s")

        return loss


class WarmRestartCosineSchedule:
    """Warm-restart cosine annealing with exponential period doubling."""

    def __init__(self, total_epochs: int, start_epoch: int = 0,
                 warmup_frac: float = 0.05):
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch
        self.warmup_frac = warmup_frac
        self.t_0 = max(total_epochs // 5, 50)

    def get_factor(self, epoch: int) -> float:
        warmup = min(epoch / max(self.warmup_frac * self.total_epochs, 1), 1.0)
        cosine = self._warm_restart_cosine(epoch)
        return warmup * cosine

    def _warm_restart_cosine(self, epoch: int) -> float:
        e = max(epoch - self.start_epoch, 0)
        t_cur = self.t_0
        consumed = 0
        while True:
            if consumed + t_cur >= e:
                progress = (e - consumed) / max(t_cur, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            consumed += t_cur
            t_cur *= 2
            if t_cur == 0:
                return 0.5
