"""Flux v3 model: WHT-D with content-dependent gating + dual-timescale memory.
Faithful PyTorch port of the Rust implementation — GPU-ready."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 4       # Semantic partition features
STRIDE = 4  # SPM update stride


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

    return (x / math.sqrt(d)).reshape(*batch_shape, d)


class FluxLayer(nn.Module):
    """Single Flux v3 layer: RMSNorm -> content-gated input -> 2x(WHT+tanh)
    -> dual-timescale EMA memory -> SPM conditioning -> residual output."""

    def __init__(self, d: int, layer_idx: int):
        super().__init__()
        self.d = d
        self.layer_idx = layer_idx
        self.res_scale = 1.0 / math.log(layer_idx + 2)

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

        # a_bias: diagonal pattern a_bias[byte, k] = 0.3 if byte == k % 256
        with torch.no_grad():
            self.a_bias.weight.zero_()
            for k in range(d):
                self.a_bias.weight[k % 256, k] = 0.3

        nn.init.normal_(self.w_gate_h, std=0.1)

        sc = 0.3 / math.sqrt(d)
        nn.init.normal_(self.c_out_fast, std=sc)
        nn.init.normal_(self.c_out_slow, std=sc)

        # delta_fast: linearly spaced 0.1 .. 0.5
        with torch.no_grad():
            frac = torch.linspace(0, 1, d)
            self.delta_fast.copy_(0.1 + frac * 0.4)
            # delta_slow: linearly spaced -2.0 .. 0.5
            self.delta_slow.copy_(-2.0 + frac * 2.5)

        # SPM delta: linearly spaced -5.0 .. -3.0
        with torch.no_grad():
            frac_k = torch.linspace(0, 1, K)
            self.spm_delta.copy_(-5.0 + frac_k * 2.0)

    def forward(self, x: torch.Tensor, byte_ids: torch.Tensor,
                spm_w: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d), byte_ids: (B, T) -> (B, T, d)"""
        B, T, d = x.shape
        device = x.device
        dtype = x.dtype

        # Pre-compute decay constants
        lam_fast = torch.exp(-F.softplus(self.delta_fast))       # (d,)
        lam_slow = torch.exp(-F.softplus(self.delta_slow))       # (d,)
        lam_sem = torch.exp(-F.softplus(self.spm_delta))         # (K,)
        gate_sig = torch.sigmoid(self.spm_gate)                  # (d,)

        h_fast = x.new_zeros(B, d)
        h_slow = x.new_zeros(B, d)
        h_sem = x.new_zeros(B, K)
        cond = x.new_zeros(B, d)

        outputs = []
        for t in range(T):
            state = x[:, t, :]       # (B, d)
            bi = byte_ids[:, t]      # (B,)

            # RMSNorm
            rms = torch.sqrt(state.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
            state = state / rms * self.rn_gamma

            # Content-gated input
            gate_in = self.g_gate(bi) + self.w_gate_h * h_fast   # (B, d)
            gate = torch.sigmoid(gate_in)
            state = gate * state + self.a_bias(bi)

            # WHT + tanh block 1
            state = wht(state)
            state = torch.tanh(self.s1 * state + self.b1)

            # WHT + tanh block 2
            state = wht(state)
            state = torch.tanh(self.s2 * state + self.b2)

            # Update dual-timescale memory
            h_fast = lam_fast * h_fast + self.b_in_fast * state
            h_slow = lam_slow * h_slow + self.b_in_slow * state

            # SPM every STRIDE steps
            if t % STRIDE == 0:
                z = h_slow @ spm_w.T                              # (B, K)
                h_sem = lam_sem * h_sem + (1 - lam_sem) * z
                cond = h_sem @ spm_w                              # (B, d)
                cond = cond * gate_sig

            # Output with residual
            out = (self.c_out_fast * h_fast
                   + self.c_out_slow * h_slow
                   + self.skip * state
                   + self.res_scale * x[:, t, :]
                   + cond)
            outputs.append(out)

        return torch.stack(outputs, dim=1)


class FluxModel(nn.Module):
    """Flux v3 byte-level language model."""

    def __init__(self, d: int = 256, n_layers: int = 3):
        super().__init__()
        assert d > 0 and (d & (d - 1)) == 0, "d must be power of 2"
        self.d = d
        self.n_layers = n_layers

        self.embedding = nn.Embedding(256, d)
        self.spm_w = nn.Parameter(torch.empty(K, d))   # Shared across layers
        self.layers = nn.ModuleList([
            FluxLayer(d, li) for li in range(n_layers)
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
        x = self.embedding(byte_ids)                   # (B, T, d)

        for layer in self.layers:
            x = layer(x, byte_ids, self.spm_w)

        logits = F.linear(x, self.head_w, self.head_b) # (B, T, 256)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, 256), targets.view(-1),
            )
            return logits, loss
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
