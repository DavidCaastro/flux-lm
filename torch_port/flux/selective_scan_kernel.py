"""Fused CUDA kernel for variable-decay parallel scan (selective scan).

Implements the general associative scan operator:
    (a1, b1) + (a2, b2) = (a2*a1, a2*b1 + b2)

where a = decay (per-timestep), b = input x.
Result: y[t] = decay[t] * y[t-1] + x[t]

This is the content-dependent extension: decay varies per (batch, time, dim),
unlike the constant-decay scan in kernels.py where decay is (dim,) only.
"""

import os
import time
import torch
import torch.nn.functional as F

# ── Logging (matches kernels.py pattern) ─────────────────────────────
_kernel_call_count = {'vscan': 0}
_KERNEL_LOG_UNTIL = 5


def _klog(msg):
    t = time.time()
    ts = time.strftime('%H:%M:%S', time.localtime(t))
    ms = int((t % 1) * 1000)
    print(f'[{ts}.{ms:03d}] [VSCAN] {msg}', flush=True)


# ═════════════════════════════════════════════════════════════════════
# CUDA source code
# ═════════════════════════════════════════════════════════════════════

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ── Variable-decay parallel scan (Hillis-Steele, associative) ──────
// Layout: decay(BD, T), x(BD, T) — pre-transposed
// Operator: (a1,b1) + (a2,b2) = (a2*a1, a2*b1 + b2)
// All arithmetic in float32; load/store in original dtype.

template<typename scalar_t>
__global__ void variable_scan_kernel(
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y_out,
    scalar_t* __restrict__ a_out,
    int BD, int T
) {
    // Shared memory: [0..T-1] = a (cumulative decay), [T..2T-1] = b (cumulative value)
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + T;

    int bd = blockIdx.x;
    int tid = threadIdx.x;
    if (bd >= BD || tid >= T) return;

    int idx = bd * T + tid;
    sa[tid] = static_cast<float>(decay[idx]);
    sb[tid] = static_cast<float>(x[idx]);
    __syncthreads();

    // Hillis-Steele: log2(T) stages
    for (int stride = 1; stride < T; stride *= 2) {
        float a_cur = sa[tid];
        float b_cur = sb[tid];
        float a_prev, b_prev;
        if (tid >= stride) {
            a_prev = sa[tid - stride];
            b_prev = sb[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            // (a_prev, b_prev) + (a_cur, b_cur) = (a_cur*a_prev, a_cur*b_prev + b_cur)
            sb[tid] = a_cur * b_prev + b_cur;
            sa[tid] = a_cur * a_prev;
        }
        __syncthreads();
    }

    y_out[idx] = static_cast<scalar_t>(sb[tid]);
    a_out[idx] = static_cast<scalar_t>(sa[tid]);
}

// ── C++ dispatch ───────────────────────────────────────────────────

#define CUDA_CHECK_LAST_ERROR(msg) do {           \
    cudaError_t err = cudaGetLastError();          \
    TORCH_CHECK(err == cudaSuccess,                \
        msg, ": ", cudaGetErrorString(err));        \
} while(0)

#define MAX_THREADS_PER_BLOCK 1024

std::vector<torch::Tensor> variable_scan_cuda(
    torch::Tensor decay, torch::Tensor x
) {
    // decay: (BD, T), x: (BD, T)
    int BD = x.size(0);
    int T = x.size(1);
    TORCH_CHECK(T <= MAX_THREADS_PER_BLOCK,
        "variable_scan_cuda: T=", T, " exceeds max threads per block (",
        MAX_THREADS_PER_BLOCK, "). Use PyTorch fallback for seq_len > 1024.");

    auto y = torch::empty_like(x);
    auto a_out = torch::empty_like(x);
    int smem_bytes = 2 * T * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "variable_scan_cuda", [&] {
            variable_scan_kernel<scalar_t><<<BD, T, smem_bytes>>>(
                decay.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                a_out.data_ptr<scalar_t>(),
                BD, T
            );
        });
    CUDA_CHECK_LAST_ERROR("variable_scan_cuda kernel launch");

    return {y, a_out};
}
"""

_CPP_SRC = r"""
std::vector<torch::Tensor> variable_scan_cuda(torch::Tensor decay, torch::Tensor x);
"""

# ═════════════════════════════════════════════════════════════════════
# JIT compilation
# ═════════════════════════════════════════════════════════════════════

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module

    _klog("JIT compilando selective scan kernel (esto puede tardar ~30s)...")
    t0 = time.time()

    from torch.utils.cpp_extension import load_inline

    build_dir = os.path.join(os.path.dirname(__file__), '..', '.selective_scan_cache')
    os.makedirs(build_dir, exist_ok=True)
    _klog(f"  build_dir={os.path.abspath(build_dir)}")

    try:
        _module = load_inline(
            name='flux_selective_scan',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['variable_scan_cuda'],
            verbose=False,
            extra_cuda_cflags=[
                '-O3', '--use_fast_math',
                '-gencode', 'arch=compute_89,code=sm_89',
                '-gencode', 'arch=compute_89,code=compute_89',
            ],
            build_directory=build_dir,
        )
        dt = time.time() - t0
        _klog(f"  Selective scan kernel compilado OK en {dt:.1f}s")
    except Exception as e:
        dt = time.time() - t0
        _klog(f"  FALLO compilacion selective scan kernel tras {dt:.1f}s: {e}")
        raise

    return _module


# ═════════════════════════════════════════════════════════════════════
# Autograd Function
# ═════════════════════════════════════════════════════════════════════

class _VariableScanFunction(torch.autograd.Function):
    """Fused variable-decay parallel scan: y[t] = decay[t]*y[t-1] + x[t].

    Forward: associative scan with operator (a2*a1, a2*b1+b2).
    Backward:
      - grad_x: reverse scan using decay[t+1] (shifted), NOT decay[t]
      - grad_decay[t] = grad_x[t] * y[t-1]  (computed in float32)
    NEVER returns None for any gradient (lesson from bug 995e7f1).
    """

    @staticmethod
    def forward(ctx, decay, x):
        """decay: (B, T, d), x: (B, T, d) -> y: (B, T, d)."""
        B, T, D = x.shape

        cc = _kernel_call_count
        cc['vscan'] += 1
        log = cc['vscan'] <= _KERNEL_LOG_UNTIL
        if log:
            _klog(f"VSCAN forward #{cc['vscan']}: B={B} T={T} D={D}, dtype={x.dtype}")

        t0 = time.time()
        mod = _get_module()

        # Transpose to (B*D, T) layout expected by kernel
        decay_t = decay.transpose(1, 2).contiguous().reshape(B * D, T)
        x_t = x.transpose(1, 2).contiguous().reshape(B * D, T)

        if log:
            _klog(f"VSCAN #{cc['vscan']}: transpose OK, shapes: "
                  f"decay_t={list(decay_t.shape)}, x_t={list(x_t.shape)}")

        t1 = time.time()
        y_t, a_cumul_t = mod.variable_scan_cuda(decay_t, x_t)
        if log:
            _klog(f"VSCAN #{cc['vscan']}: kernel OK ({time.time()-t1:.4f}s)")

        y = y_t.reshape(B, D, T).transpose(1, 2).contiguous()
        if log:
            _klog(f"VSCAN #{cc['vscan']}: reshape OK, total={time.time()-t0:.4f}s")

        ctx.save_for_backward(decay, y)
        ctx.shape = (B, T, D)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        decay, y = ctx.saved_tensors
        B, T, D = ctx.shape
        mod = _get_module()

        # Reverse recurrence: grad_x[t] = decay[t+1] * grad_x[t+1] + grad_output[t]
        # Note: uses decay[t+1] (NEXT timestep), not decay[t].
        # decay_next[t] = decay[t+1] for t<T-1, decay_next[T-1] = 0
        decay_next = F.pad(decay[:, 1:, :], (0, 0, 0, 1))  # shift left, pad 0 at end
        decay_next_flip = decay_next.flip(1)
        grad_flip = grad_output.flip(1)

        grad_flip_t = grad_flip.transpose(1, 2).contiguous().reshape(B * D, T)
        decay_next_flip_t = decay_next_flip.transpose(1, 2).contiguous().reshape(B * D, T)

        grad_scan_t, _ = mod.variable_scan_cuda(decay_next_flip_t, grad_flip_t)

        grad_x = grad_scan_t.reshape(B, D, T).transpose(1, 2).contiguous().flip(1)

        # grad_decay[t] = grad_x[t] * y[t-1], computed in float32 to avoid bf16 overflow
        y_prev = F.pad(y[:, :-1, :], (0, 0, 1, 0))
        grad_decay = (grad_x.float() * y_prev.float()).to(grad_output.dtype)

        return grad_decay, grad_x


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════

def variable_parallel_scan_fused(decay: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Fused variable-decay parallel scan.

    decay: (B, T, d) — per-timestep decay values
    x: (B, T, d) — input sequence

    Falls back to PyTorch if T > 1024 (CUDA max threads per block).
    """
    T = x.shape[1]
    if T > 1024:
        _klog(f"VSCAN fallback: T={T} > 1024, usando PyTorch")
        from .model import parallel_scan
        return parallel_scan(decay, x)
    return _VariableScanFunction.apply(decay, x)


# Flag for external checks
HAS_SELECTIVE_SCAN_KERNEL = True
