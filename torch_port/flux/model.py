"""Flux v3 model: WHT-D with content-dependent gating + dual-timescale memory.
Faithful PyTorch port of the Rust implementation — GPU-ready.

Supports two execution modes:
  - Sequential (parallel=False): exact, O(T) per layer — reference implementation
  - Parallel  (parallel=True):  mean-field + perturbative correction,
    O(T log T) work / O(log T) depth via associative scan
"""

import math
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
    """Parallel linear recurrence: y[t] = decay · y[t-1] + x[t].

    For constant decay, uses recursive doubling:
      Round k: y[t] += decay^(2^k) · y[t - 2^k]
    After ⌈log₂ T⌉ rounds, y[t] = Σ_{k=0}^{t} decay^k · x[t-k].

    Mathematically identical to the sequential scan.
    O(T log T) work, O(log T) depth — fully GPU-parallel.

    Args:
        decay: (D,) constant decay per dimension
        x:     (B, T, D) input sequence
    Returns:
        y:     (B, T, D) accumulated states
    """
    y = x
    stride = 1
    decay_pow = decay.view(1, 1, -1)   # (1, 1, D) — broadcast over B, T

    while stride < y.shape[1]:
        # y_shifted[t] = y[t - stride] * decay^stride,  zero-padded on left
        y_shifted = F.pad(y[:, :-stride, :], (0, 0, stride, 0)) * decay_pow
        y = y + y_shifted
        decay_pow = decay_pow * decay_pow   # decay^(2^k) → decay^(2^(k+1))
        stride *= 2

    return y


# ═══════════════════════════════════════════════════════════════════════
# FluxLayer
# ═══════════════════════════════════════════════════════════════════════

class FluxLayer(nn.Module):
    """Single Flux v3 layer: RMSNorm → content-gated input → 2×(WHT+tanh)
    → dual-timescale EMA memory → SPM conditioning → residual output.

    Args:
        d:              model dimension (power of 2)
        layer_idx:      0-indexed layer number
        parallel:       use parallel scan instead of sequential loop
        n_corrections:  perturbative correction steps (0 = pure mean-field,
                        1 = one correction, default; higher = more accurate)
    """

    def __init__(self, d: int, layer_idx: int,
                 parallel: bool = False, n_corrections: int = 1,
                 use_fused: bool = True):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        self.res_scale = 1.0 / math.log(layer_idx + 2)
        self.parallel = parallel
        self.n_corrections = n_corrections
        self.use_fused = use_fused and HAS_FUSED_KERNELS

        # RMSNorm
        self.rn_gamma = nn.Parameter(torch.ones(d))

        # Content-gated input (byte-indexed)
        self.g_gate = nn.Embedding(256, d)
        self.a_bias = nn.Embedding(256, d)
        self.w_gate_h = nn.Parameter(torch.empty(d))

        # WHT blocks
        self.s1 = nn.Parameter(torch.ones(d))
        self.b1 = nn.Parameter(torch.zeros(d))
        self.s2 = nn.Parameter(torch.ones(d))
        self.b2 = nn.Parameter(torch.zeros(d))

        # Dual-timescale memory (delta parametrizes decay via softplus)
        self.delta_fast = nn.Parameter(torch.empty(d))
        self.b_in_fast = nn.Parameter(torch.ones(d))
        self.c_out_fast = nn.Parameter(torch.empty(d))
        self.skip = nn.Parameter(torch.full((d,), 0.5))
        self.delta_slow = nn.Parameter(torch.empty(d))
        self.b_in_slow = nn.Parameter(torch.ones(d))
        self.c_out_slow = nn.Parameter(torch.empty(d))

        # SPM per-layer params
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
        """Reference implementation: step-by-step loop."""
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
        """Mean-field decomposition with iterative perturbative correction.

        The sequential bottleneck is:
            state[t] = φ(x[t]) · σ(g[t] + w · h_fast[t-1]) + a[t]
            h_fast[t] = λ · h_fast[t-1] + β · state[t]

        state depends on h_fast (nonlinear feedback via sigmoid gate).
        We decouple this by iterating:

          Order 0:  gate⁰ = σ(g)                           (no coupling)
                    state⁰ = gate⁰ · x_norm + a            (parallel ∀t)
                    h_fast⁰ = parallel_scan(λ, β·state⁰)   (O(log T))

          Order k:  gateᵏ = σ(g + w · h_fastᵏ⁻¹[t-1])     (parallel ∀t)
                    stateᵏ = gateᵏ · x_norm + a             (parallel ∀t)
                    h_fastᵏ = parallel_scan(λ, β·stateᵏ)    (O(log T))

        Each correction refines the gate using the previous scan's output.
        Convergence is fast because w·h_fast is a small perturbation on g.
        Residual error after k corrections is O(||w||^(k+1)).
        """
        B, T, d = x.shape

        lam_fast = torch.exp(-F.softplus(self.delta_fast))   # (d,)
        lam_slow = torch.exp(-F.softplus(self.delta_slow))   # (d,)
        lam_sem = torch.exp(-F.softplus(self.spm_delta))     # (K,)
        gate_sig = torch.sigmoid(self.spm_gate)              # (d,)

        # ── All-positions RMSNorm (parallel) ──
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
        x_norm = x / rms * self.rn_gamma                    # (B, T, d)

        # ── All-positions byte lookups (parallel) ──
        g_all = self.g_gate(byte_ids)                        # (B, T, d)
        a_all = self.a_bias(byte_ids)                        # (B, T, d)

        # ── Iterative mean-field refinement ──
        h_fast = None

        for _ in range(self.n_corrections + 1):
            # Gate: order 0 uses no coupling, order k uses h_fast^(k-1)
            if h_fast is None:
                gate = torch.sigmoid(g_all)
            else:
                # h_fast[t-1] for gate at position t: shift right, pad zero
                h_prev = F.pad(h_fast[:, :-1, :], (0, 0, 1, 0))
                gate = torch.sigmoid(g_all + self.w_gate_h * h_prev)

            # State transform (all positions parallel)
            state = gate * x_norm + a_all                    # (B, T, d)
            if self.use_fused and state.is_cuda:
                state = wht_scale_tanh_fused(state, self.s1, self.b1)
                state = wht_scale_tanh_fused(state, self.s2, self.b2)
            else:
                state = torch.tanh(self.s1 * wht(state) + self.b1)
                state = torch.tanh(self.s2 * wht(state) + self.b2)

            # Parallel scan for both timescales
            if self.use_fused and state.is_cuda:
                h_fast = parallel_scan_fused(lam_fast, self.b_in_fast * state)
                h_slow = parallel_scan_fused(lam_slow, self.b_in_slow * state)
            else:
                h_fast = parallel_scan(lam_fast, self.b_in_fast * state)
                h_slow = parallel_scan(lam_slow, self.b_in_slow * state)

        # ── SPM (parallel) ──
        n_sub = (T + STRIDE - 1) // STRIDE
        idx = torch.arange(0, T, STRIDE, device=x.device)[:n_sub]
        h_slow_sub = h_slow[:, idx, :]                       # (B, n_sub, d)

        z_sub = h_slow_sub @ spm_w.T                         # (B, n_sub, K)
        if self.use_fused and z_sub.is_cuda:
            h_sem = parallel_scan_fused(lam_sem, (1 - lam_sem) * z_sub)
        else:
            h_sem = parallel_scan(lam_sem, (1 - lam_sem) * z_sub)
        cond_sub = (h_sem @ spm_w) * gate_sig                # (B, n_sub, d)

        # Broadcast cond to full resolution (hold value between updates)
        cond = cond_sub.repeat_interleave(STRIDE, dim=1)[:, :T, :]

        # ── Output (all positions parallel) ──
        return (self.c_out_fast * h_fast
                + self.c_out_slow * h_slow
                + self.skip * state
                + self.res_scale * x
                + cond)


# ═══════════════════════════════════════════════════════════════════════
# FluxModel
# ═══════════════════════════════════════════════════════════════════════

class FluxModel(nn.Module):
    """Flux v3 byte-level language model.

    Args:
        d:              embedding dimension (must be power of 2)
        n_layers:       number of Flux layers
        parallel:       use parallel scan (mean-field) for all layers
        n_corrections:  perturbative correction depth (default 1)
    """

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
        """byte_ids: (B, T), targets: (B, T) optional -> logits (B, T, 256)"""
        x = self.embedding(byte_ids)

        for layer in self.layers:
            x = layer(x, byte_ids, self.spm_w)

        logits = F.linear(x, self.head_w, self.head_b)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1))
            return logits, loss
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_mode(self, parallel: bool, n_corrections: int = 1,
                 use_fused: bool = True):
        """Switch all layers between sequential and parallel mode."""
        for layer in self.layers:
            layer.parallel = parallel
            layer.n_corrections = n_corrections
            layer.use_fused = use_fused and HAS_FUSED_KERNELS
