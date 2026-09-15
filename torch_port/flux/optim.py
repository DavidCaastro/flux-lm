"""EntropicAdam — Adam with sign-entropy-based per-group learning rate scaling.
Faithful port of the Rust EntropicAdam optimizer."""

import math
import torch
from torch.optim import Optimizer


class EntropicAdam(Optimizer):
    """Adam optimizer with entropic learning rate scaling.

    For each group of `group_size` parameters, tracks the majority sign
    direction of gradients over a 32-step history. The binary entropy H of
    the sign ratio modulates the learning rate: lr_group = lr * exp(-H / T),
    where T anneals from t_initial to t_final over training.

    High entropy (random sign flips) -> lower lr (cautious).
    Low entropy (consistent direction) -> higher lr (confident).
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
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            gs = group['group_size']
            t_initial = group['t_initial']
            t_final = group['t_final']
            total_epochs = group['total_epochs']

            frac = epoch / total_epochs
            t_epoch = t_initial * (t_final / t_initial) ** frac

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
                    state['sign_history'] = [0] * n_groups

                state['step'] += 1
                t = state['step']
                m, v = state['m'], state['v']
                sign_hist = state['sign_history']

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                lr_base = lr * math.sqrt(bc2) / bc1

                # Update moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute per-group sign entropy LR
                flat_grad = grad.reshape(-1)
                flat_m = m.reshape(-1)
                flat_v = v.reshape(-1)
                flat_p = p.reshape(-1)
                n = flat_grad.numel()
                n_groups = len(sign_hist)

                for g in range(n_groups):
                    start = g * gs
                    end = min(start + gs, n)
                    chunk = flat_grad[start:end]

                    pos = (chunk > 0).sum().item()
                    majority = 1 if pos * 2 >= (end - start) else 0

                    sign_hist[g] = ((sign_hist[g] << 1) & 0xFFFFFFFF) | majority

                    bits = bin(sign_hist[g] & 0xFFFFFFFF).count('1')
                    p_ratio = bits / 32.0
                    if p_ratio < 1e-10 or p_ratio > 1.0 - 1e-10:
                        h = 0.0
                    else:
                        h = -(p_ratio * math.log(p_ratio)
                              + (1 - p_ratio) * math.log(1 - p_ratio))

                    glr = lr_base * math.exp(-h / t_epoch)

                    # Apply Adam update for this group
                    m_chunk = flat_m[start:end]
                    v_chunk = flat_v[start:end]
                    flat_p[start:end] -= glr * m_chunk / (v_chunk.sqrt() + eps)

        return loss


class WarmRestartCosineSchedule:
    """Warm-restart cosine annealing with exponential period doubling.
    Matches the Rust warm_restart_cosine function."""

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
