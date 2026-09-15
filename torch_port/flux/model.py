"""Flux v3 model — INSTRUMENTED: ultra-verbose logging for debugging hangs.

Logs BEFORE every operation so the last log line reveals where a hang occurs.
First N forward passes log every sub-operation with tensor stats.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .kernels import wht_fused, wht_scale_tanh_fused, parallel_scan_fused
    HAS_FUSED_KERNELS = True
except (ImportError, RuntimeError):
    HAS_FUSED_KERNELS = False

K = 4       # Semantic partition features
STRIDE = 4  # SPM update stride
_MAX_FUSED_DIM = 1024  # CUDA max threads per block

# ── Debug logging control ─────────────────────────────────────────────
_model_fwd_count = 0
_VERBOSE_UNTIL = 12     # Full tensor stats for first N forward passes
_LAYER_LOG_UNTIL = 20   # Layer entry/exit logging for first N passes


def reset_debug_counters():
    """Reset forward pass counter (call after preflight)."""
    global _model_fwd_count
    _model_fwd_count = 0


def _tlog(msg):
    t = time.time()
    ts = time.strftime('%H:%M:%S', time.localtime(t))
    ms = int((t % 1) * 1000)
    print(f'[{ts}.{ms:03d}] {msg}', flush=True)


def _tstat(name, t):
    """One-line tensor stats. Triggers implicit GPU sync via .item()."""
    if t is None:
        return f"{name}=None"
    with torch.no_grad():
        ft = t.float()
        n = ft.norm().item()
        mn = t.min().item()
        mx = t.max().item()
        has_nan = t.isnan().any().item()
        has_inf = t.isinf().any().item()
    return (f"{name}: shape={list(t.shape)} norm={n:.4f} "
            f"min={mn:.4f} max={mx:.4f} nan={has_nan} inf={has_inf}")


# ═══════════════════════════════════════════════════════════════════════
# Core primitives
# ═══════════════════════════════════════════════════════════════════════

def wht(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard Transform on last dimension. d must be power of 2."""
    d = x.shape[-1]
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, d)

    half = 1
    while half < d:
        x = x.view(-1, d // (2 * half), 2, half)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2).reshape(-1, d)
        half *= 2

    return (x * (1.0 / math.sqrt(d))).reshape(*batch_shape, d)


def parallel_scan(decay: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Parallel linear recurrence: y[t] = decay * y[t-1] + x[t]."""
    y = x
    stride = 1
    decay_pow = decay.view(1, 1, -1)

    while stride < y.shape[1]:
        y_shifted = F.pad(y[:, :-stride, :], (0, 0, stride, 0)) * decay_pow
        y = y + y_shifted
        decay_pow = decay_pow * decay_pow
        stride *= 2

    return y


# ═══════════════════════════════════════════════════════════════════════
# FluxLayer
# ═══════════════════════════════════════════════════════════════════════

class FluxLayer(nn.Module):
    """Single Flux v3 layer with instrumented logging."""

    def __init__(self, d: int, layer_idx: int,
                 parallel: bool = False, n_corrections: int = 1,
                 use_fused: bool = True):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        self.res_scale = 1.0 / math.log(layer_idx + 2)
        self.parallel = parallel
        self.n_corrections = n_corrections
        self.use_fused = use_fused and HAS_FUSED_KERNELS and d <= _MAX_FUSED_DIM

        self.rn_gamma = nn.Parameter(torch.ones(d))
        self.g_gate = nn.Embedding(256, d)
        self.a_bias = nn.Embedding(256, d)
        self.w_gate_h = nn.Parameter(torch.empty(d))
        self.s1 = nn.Parameter(torch.ones(d))
        self.b1 = nn.Parameter(torch.zeros(d))
        self.s2 = nn.Parameter(torch.ones(d))
        self.b2 = nn.Parameter(torch.zeros(d))
        self.delta_fast = nn.Parameter(torch.empty(d))
        self.b_in_fast = nn.Parameter(torch.ones(d))
        self.c_out_fast = nn.Parameter(torch.empty(d))
        self.skip = nn.Parameter(torch.full((d,), 0.5))
        self.delta_slow = nn.Parameter(torch.empty(d))
        self.b_in_slow = nn.Parameter(torch.ones(d))
        self.c_out_slow = nn.Parameter(torch.empty(d))
        self.spm_delta = nn.Parameter(torch.empty(K))
        self.spm_gate = nn.Parameter(torch.full((d,), -5.0))
        self._init_params()

    def _init_params(self):
        d = self.d
        nn.init.normal_(self.g_gate.weight, std=0.1)
        with torch.no_grad():
            self.a_bias.weight.zero_()
            for k in range(d):
                self.a_bias.weight[k % 256, k] = 0.3
        nn.init.normal_(self.w_gate_h, std=0.1)
        sc = 0.3 / math.sqrt(d)
        nn.init.normal_(self.c_out_fast, std=sc)
        nn.init.normal_(self.c_out_slow, std=sc)
        with torch.no_grad():
            frac = torch.linspace(0, 1, d)
            self.delta_fast.copy_(0.1 + frac * 0.4)
            self.delta_slow.copy_(-2.0 + frac * 2.5)
        with torch.no_grad():
            frac_k = torch.linspace(0, 1, K)
            self.spm_delta.copy_(-5.0 + frac_k * 2.0)

    def forward(self, x: torch.Tensor, byte_ids: torch.Tensor,
                spm_w: torch.Tensor) -> torch.Tensor:
        if self.parallel:
            return self._forward_parallel(x, byte_ids, spm_w)
        return self._forward_sequential(x, byte_ids, spm_w)

    # ── Sequential (exact, O(T)) ──────────────────────────────────────

    def _forward_sequential(self, x: torch.Tensor, byte_ids: torch.Tensor,
                            spm_w: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        lam_fast = torch.exp(-F.softplus(self.delta_fast))
        lam_slow = torch.exp(-F.softplus(self.delta_slow))
        lam_sem = torch.exp(-F.softplus(self.spm_delta))
        gate_sig = torch.sigmoid(self.spm_gate)
        h_fast = x.new_zeros(B, d)
        h_slow = x.new_zeros(B, d)
        h_sem = x.new_zeros(B, K)
        cond = x.new_zeros(B, d)
        outputs = []
        for t in range(T):
            state = x[:, t, :]
            bi = byte_ids[:, t]
            rms = torch.sqrt(state.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            state = state / rms * self.rn_gamma
            gate_in = self.g_gate(bi) + self.w_gate_h * h_fast
            gate = torch.sigmoid(gate_in)
            state = gate * state + self.a_bias(bi)
            state = torch.tanh(self.s1 * wht(state) + self.b1)
            state = torch.tanh(self.s2 * wht(state) + self.b2)
            h_fast = lam_fast * h_fast + self.b_in_fast * state
            h_slow = lam_slow * h_slow + self.b_in_slow * state
            if t % STRIDE == 0:
                z = h_slow @ spm_w.T
                h_sem = lam_sem * h_sem + (1 - lam_sem) * z
                cond = (h_sem @ spm_w) * gate_sig
            out = (self.c_out_fast * h_fast
                   + self.c_out_slow * h_slow
                   + self.skip * state
                   + self.res_scale * x[:, t, :]
                   + cond)
            outputs.append(out)
        return torch.stack(outputs, dim=1)

    # ── Parallel (mean-field + perturbative, O(T log T)) ──────────────

    def _forward_parallel(self, x: torch.Tensor, byte_ids: torch.Tensor,
                          spm_w: torch.Tensor) -> torch.Tensor:
        global _model_fwd_count
        verbose = _model_fwd_count <= _VERBOSE_UNTIL
        li = self.layer_idx
        B, T, d = x.shape

        if verbose:
            _tlog(f"      L{li}: START parallel — B={B} T={T} d={d} "
                  f"fused={self.use_fused} n_corr={self.n_corrections}")
            _tlog(f"      L{li}: {_tstat('x_in', x)}")

        # ── Decay lambdas ──
        if verbose:
            _tlog(f"      L{li}: computing decay lambdas...")
        t0 = time.time()
        lam_fast = torch.exp(-F.softplus(self.delta_fast))
        lam_slow = torch.exp(-F.softplus(self.delta_slow))
        lam_sem = torch.exp(-F.softplus(self.spm_delta))
        gate_sig = torch.sigmoid(self.spm_gate)
        if verbose:
            _tlog(f"      L{li}: lambdas OK ({time.time()-t0:.4f}s)")
            _tlog(f"      L{li}: {_tstat('lam_fast', lam_fast)}")
            _tlog(f"      L{li}: {_tstat('lam_slow', lam_slow)}")
            _tlog(f"      L{li}: {_tstat('lam_sem', lam_sem)}")
            _tlog(f"      L{li}: {_tstat('gate_sig', gate_sig)}")

        # ── RMSNorm ──
        if verbose:
            _tlog(f"      L{li}: RMSNorm...")
        t0 = time.time()
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        x_norm = x / rms * self.rn_gamma
        if verbose:
            _tlog(f"      L{li}: RMSNorm OK ({time.time()-t0:.4f}s)")
            _tlog(f"      L{li}: {_tstat('x_norm', x_norm)}")

        # ── Byte lookups ──
        if verbose:
            _tlog(f"      L{li}: byte lookups (g_gate, a_bias)...")
        t0 = time.time()
        g_all = self.g_gate(byte_ids)
        a_all = self.a_bias(byte_ids)
        if verbose:
            _tlog(f"      L{li}: lookups OK ({time.time()-t0:.4f}s)")
            _tlog(f"      L{li}: {_tstat('g_all', g_all)}")
            _tlog(f"      L{li}: {_tstat('a_all', a_all)}")

        # ── Iterative mean-field refinement ──
        h_fast = None

        for corr in range(self.n_corrections + 1):
            if verbose:
                _tlog(f"      L{li}: === correction {corr}/{self.n_corrections} ===")

            # Gate
            if verbose:
                _tlog(f"      L{li}: computing gate...")
            t0 = time.time()
            if h_fast is None:
                gate = torch.sigmoid(g_all)
            else:
                h_prev = F.pad(h_fast[:, :-1, :], (0, 0, 1, 0))
                gate = torch.sigmoid(g_all + self.w_gate_h * h_prev)
            if verbose:
                _tlog(f"      L{li}: gate OK ({time.time()-t0:.4f}s)")
                _tlog(f"      L{li}: {_tstat('gate', gate)}")

            # State transform
            if verbose:
                _tlog(f"      L{li}: state = gate*x_norm + a_all...")
            t0 = time.time()
            state = gate * x_norm + a_all
            if verbose:
                _tlog(f"      L{li}: state OK ({time.time()-t0:.4f}s)")
                _tlog(f"      L{li}: {_tstat('state_pre_wht', state)}")

            # WHT block 1
            if self.use_fused and state.is_cuda:
                if verbose:
                    _tlog(f"      L{li}: wht_scale_tanh_fused #1 "
                          f"(state.shape={list(state.shape)})...")
                t0 = time.time()
                state = wht_scale_tanh_fused(state, self.s1, self.b1)
                if verbose:
                    _tlog(f"      L{li}: WHT#1 fused OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('state_post_wht1', state)}")
            else:
                if verbose:
                    _tlog(f"      L{li}: PyTorch WHT #1...")
                t0 = time.time()
                state = torch.tanh(self.s1 * wht(state) + self.b1)
                if verbose:
                    _tlog(f"      L{li}: WHT#1 pytorch OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('state_post_wht1', state)}")

            # WHT block 2
            if self.use_fused and state.is_cuda:
                if verbose:
                    _tlog(f"      L{li}: wht_scale_tanh_fused #2...")
                t0 = time.time()
                state = wht_scale_tanh_fused(state, self.s2, self.b2)
                if verbose:
                    _tlog(f"      L{li}: WHT#2 fused OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('state_post_wht2', state)}")
            else:
                if verbose:
                    _tlog(f"      L{li}: PyTorch WHT #2...")
                t0 = time.time()
                state = torch.tanh(self.s2 * wht(state) + self.b2)
                if verbose:
                    _tlog(f"      L{li}: WHT#2 pytorch OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('state_post_wht2', state)}")

            # Parallel scan: h_fast
            if self.use_fused and state.is_cuda:
                if verbose:
                    _tlog(f"      L{li}: parallel_scan_fused h_fast "
                          f"(input.shape={list(state.shape)})...")
                t0 = time.time()
                h_fast = parallel_scan_fused(lam_fast, self.b_in_fast * state)
                if verbose:
                    _tlog(f"      L{li}: scan h_fast OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('h_fast', h_fast)}")
            else:
                if verbose:
                    _tlog(f"      L{li}: PyTorch scan h_fast...")
                t0 = time.time()
                h_fast = parallel_scan(lam_fast, self.b_in_fast * state)
                if verbose:
                    _tlog(f"      L{li}: scan h_fast OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('h_fast', h_fast)}")

            # Parallel scan: h_slow
            if self.use_fused and state.is_cuda:
                if verbose:
                    _tlog(f"      L{li}: parallel_scan_fused h_slow...")
                t0 = time.time()
                h_slow = parallel_scan_fused(lam_slow, self.b_in_slow * state)
                if verbose:
                    _tlog(f"      L{li}: scan h_slow OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('h_slow', h_slow)}")
            else:
                if verbose:
                    _tlog(f"      L{li}: PyTorch scan h_slow...")
                t0 = time.time()
                h_slow = parallel_scan(lam_slow, self.b_in_slow * state)
                if verbose:
                    _tlog(f"      L{li}: scan h_slow OK ({time.time()-t0:.4f}s)")
                    _tlog(f"      L{li}: {_tstat('h_slow', h_slow)}")

        # ── SPM ──
        if verbose:
            _tlog(f"      L{li}: SPM computation...")
        t0 = time.time()
        n_sub = (T + STRIDE - 1) // STRIDE
        idx = torch.arange(0, T, STRIDE, device=x.device)[:n_sub]
        h_slow_sub = h_slow[:, idx, :]
        if verbose:
            _tlog(f"      L{li}: SPM n_sub={n_sub}, "
                  f"{_tstat('h_slow_sub', h_slow_sub)}")

        z_sub = h_slow_sub @ spm_w.T
        if verbose:
            _tlog(f"      L{li}: {_tstat('z_sub', z_sub)}")

        if self.use_fused and z_sub.is_cuda:
            if verbose:
                _tlog(f"      L{li}: parallel_scan_fused h_sem "
                      f"(shape={list(z_sub.shape)})...")
            h_sem = parallel_scan_fused(lam_sem, (1 - lam_sem) * z_sub)
        else:
            if verbose:
                _tlog(f"      L{li}: PyTorch scan h_sem...")
            h_sem = parallel_scan(lam_sem, (1 - lam_sem) * z_sub)
        if verbose:
            _tlog(f"      L{li}: {_tstat('h_sem', h_sem)}")

        cond_sub = (h_sem @ spm_w) * gate_sig
        cond = cond_sub.repeat_interleave(STRIDE, dim=1)[:, :T, :]
        if verbose:
            dt = time.time() - t0
            _tlog(f"      L{li}: SPM OK ({dt:.4f}s), {_tstat('cond', cond)}")

        # ── Output ──
        if verbose:
            _tlog(f"      L{li}: computing output...")
        t0 = time.time()
        out = (self.c_out_fast * h_fast
               + self.c_out_slow * h_slow
               + self.skip * state
               + self.res_scale * x
               + cond)
        if verbose:
            _tlog(f"      L{li}: output OK ({time.time()-t0:.4f}s)")
            _tlog(f"      L{li}: {_tstat('out', out)}")
            _tlog(f"      L{li}: DONE")

        return out


# ═══════════════════════════════════════════════════════════════════════
# FluxModel
# ═══════════════════════════════════════════════════════════════════════

class FluxModel(nn.Module):
    """Flux v3 byte-level language model with instrumented logging."""

    def __init__(self, d: int = 256, n_layers: int = 3,
                 parallel: bool = False, n_corrections: int = 1,
                 use_fused: bool = True):
        super().__init__()
        assert d > 0 and (d & (d - 1)) == 0, "d must be power of 2"
        self.d = d
        self.n_layers = n_layers

        self.embedding = nn.Embedding(256, d)
        self.spm_w = nn.Parameter(torch.empty(K, d))
        self.layers = nn.ModuleList([
            FluxLayer(d, li, parallel=parallel, n_corrections=n_corrections,
                      use_fused=use_fused)
            for li in range(n_layers)
        ])
        self.head_w = nn.Parameter(torch.empty(256, d))
        self.head_b = nn.Parameter(torch.zeros(256))

        self._init_params()

    def _init_params(self):
        d = self.d
        nn.init.normal_(self.embedding.weight, std=0.1)
        xavier = math.sqrt(2.0 / (K + d))
        nn.init.normal_(self.spm_w, std=xavier)
        he = math.sqrt(2.0 / (d + 256))
        nn.init.normal_(self.head_w, std=he)

    def forward(self, byte_ids: torch.Tensor,
                targets: torch.Tensor | None = None):
        global _model_fwd_count
        _model_fwd_count += 1
        cnt = _model_fwd_count
        verbose = cnt <= _VERBOSE_UNTIL
        layer_log = cnt <= _LAYER_LOG_UNTIL

        if verbose:
            _tlog(f"    FluxModel.forward #{cnt}: "
                  f"byte_ids={list(byte_ids.shape)}, "
                  f"targets={'yes' if targets is not None else 'no'}, "
                  f"device={byte_ids.device}")

        # Embedding
        if verbose:
            _tlog(f"    #{cnt}: embedding lookup...")
        t0 = time.time()
        x = self.embedding(byte_ids)
        if verbose:
            _tlog(f"    #{cnt}: embedding OK ({time.time()-t0:.4f}s), "
                  f"{_tstat('x_emb', x)}")

        # Layers
        for i, layer in enumerate(self.layers):
            if layer_log:
                _tlog(f"    #{cnt}: Layer {i}/{self.n_layers} entering...")
            t0 = time.time()
            x = layer(x, byte_ids, self.spm_w)
            dt = time.time() - t0
            if layer_log:
                _tlog(f"    #{cnt}: Layer {i}/{self.n_layers} done "
                      f"({dt:.4f}s), out_norm={x.float().norm().item():.4f}")

        # Head
        if verbose:
            _tlog(f"    #{cnt}: logits = F.linear(x, head_w, head_b)...")
        t0 = time.time()
        logits = F.linear(x, self.head_w, self.head_b)
        if verbose:
            _tlog(f"    #{cnt}: logits OK ({time.time()-t0:.4f}s), "
                  f"{_tstat('logits', logits)}")

        if targets is not None:
            if verbose:
                _tlog(f"    #{cnt}: cross_entropy loss...")
            t0 = time.time()
            loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1))
            if verbose:
                _tlog(f"    #{cnt}: loss={loss.item():.6f} ({time.time()-t0:.4f}s)")
                _tlog(f"    #{cnt}: forward DONE")
            return logits, loss

        if verbose:
            _tlog(f"    #{cnt}: forward DONE (no loss)")
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_mode(self, parallel: bool, n_corrections: int = 1,
                 use_fused: bool = True):
        for layer in self.layers:
            layer.parallel = parallel
            layer.n_corrections = n_corrections
            layer.use_fused = (use_fused and HAS_FUSED_KERNELS
                               and layer.d <= _MAX_FUSED_DIM)
