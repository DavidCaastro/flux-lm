"""Fused CUDA kernels for Flux v3 — INSTRUMENTED with logging.

JIT compilation and first N kernel calls are logged to diagnose hangs.
"""

import math
import os
import time
import torch
import torch.nn.functional as F

# ── Kernel logging ────────────────────────────────────────────────────
_kernel_call_count = {'wht': 0, 'wht_st': 0, 'scan': 0}
_KERNEL_LOG_UNTIL = 5  # Log first N calls to each kernel


def _klog(msg):
    t = time.time()
    ts = time.strftime('%H:%M:%S', time.localtime(t))
    ms = int((t % 1) * 1000)
    print(f'[{ts}.{ms:03d}] [KERNEL] {msg}', flush=True)


# ═══════════════════════════════════════════════════════════════════════
# CUDA source code
# ═══════════════════════════════════════════════════════════════════════

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

// ── WHT butterfly (pure, no scale/bias/tanh) ──────────────────────────

template<typename scalar_t>
__global__ void wht_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    int d, int log_d, int N
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N || tid >= d) return;

    smem[tid] = static_cast<float>(x[row * d + tid]);
    __syncthreads();

    for (int s = 0; s < log_d; s++) {
        int half = 1 << s;
        int group = 2 * half;
        int idx = tid % group;
        float a, b;
        if (idx < half) {
            a = smem[tid];
            b = smem[tid + half];
        } else {
            a = smem[tid - half];
            b = smem[tid];
        }
        __syncthreads();
        smem[tid] = (idx < half) ? (a + b) : (a - b);
        __syncthreads();
    }

    float scale = rsqrtf(static_cast<float>(d));
    out[row * d + tid] = static_cast<scalar_t>(smem[tid] * scale);
}

// ── WHT + scale + bias + tanh (fused) ─────────────────────────────────

template<typename scalar_t>
__global__ void wht_scale_tanh_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int d, int log_d, int N
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N || tid >= d) return;

    smem[tid] = static_cast<float>(x[row * d + tid]);
    __syncthreads();

    for (int s = 0; s < log_d; s++) {
        int half = 1 << s;
        int group = 2 * half;
        int idx = tid % group;
        float a, b;
        if (idx < half) {
            a = smem[tid];
            b = smem[tid + half];
        } else {
            a = smem[tid - half];
            b = smem[tid];
        }
        __syncthreads();
        smem[tid] = (idx < half) ? (a + b) : (a - b);
        __syncthreads();
    }

    float wht_scale = rsqrtf(static_cast<float>(d));
    float val = smem[tid] * wht_scale;
    float s = static_cast<float>(scale[tid]);
    float b = static_cast<float>(bias[tid]);
    out[row * d + tid] = static_cast<scalar_t>(tanhf(s * val + b));
}

// ── Parallel prefix scan (Hillis-Steele) ──────────────────────────────
// Input/output layout: (BD, T) where BD = B*D, pre-transposed
// decay: (D,) — constant per dimension

template<typename scalar_t>
__global__ void parallel_scan_kernel(
    const scalar_t* __restrict__ decay,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int BD, int T, int D
) {
    extern __shared__ float smem[];
    int bd = blockIdx.x;
    int tid = threadIdx.x;
    if (bd >= BD || tid >= T) return;

    int d_idx = bd % D;
    float lam = static_cast<float>(decay[d_idx]);

    // Load into shared memory
    smem[tid] = static_cast<float>(x[bd * T + tid]);
    __syncthreads();

    // Hillis-Steele with geometric decay
    float decay_pow = lam;
    for (int stride = 1; stride < T; stride *= 2) {
        float val = smem[tid];
        if (tid >= stride) {
            val += decay_pow * smem[tid - stride];
        }
        __syncthreads();
        smem[tid] = val;
        __syncthreads();
        decay_pow *= decay_pow;
    }

    y[bd * T + tid] = static_cast<scalar_t>(smem[tid]);
}

// ── C++ dispatch functions ────────────────────────────────────────────

#define CUDA_CHECK_LAST_ERROR(msg) do {           \
    cudaError_t err = cudaGetLastError();          \
    TORCH_CHECK(err == cudaSuccess,                \
        msg, ": ", cudaGetErrorString(err));        \
} while(0)

#define MAX_THREADS_PER_BLOCK 1024

torch::Tensor wht_cuda(torch::Tensor x) {
    // x: (N, d)
    int N = x.size(0);
    int d = x.size(1);
    TORCH_CHECK(d <= MAX_THREADS_PER_BLOCK,
        "wht_cuda: d=", d, " exceeds max threads per block (",
        MAX_THREADS_PER_BLOCK, "). Use PyTorch fallback for d > 1024.");
    int log_d = 0;
    for (int tmp = d; tmp > 1; tmp >>= 1) log_d++;

    auto out = torch::empty_like(x);
    int smem_bytes = d * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "wht_cuda", [&] {
            wht_kernel<scalar_t><<<N, d, smem_bytes>>>(
                x.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                d, log_d, N
            );
        });
    CUDA_CHECK_LAST_ERROR("wht_cuda kernel launch");

    return out;
}

torch::Tensor wht_scale_tanh_cuda(
    torch::Tensor x, torch::Tensor scale, torch::Tensor bias
) {
    // x: (N, d), scale: (d,), bias: (d,)
    int N = x.size(0);
    int d = x.size(1);
    TORCH_CHECK(d <= MAX_THREADS_PER_BLOCK,
        "wht_scale_tanh_cuda: d=", d, " exceeds max threads per block (",
        MAX_THREADS_PER_BLOCK, "). Use PyTorch fallback for d > 1024.");
    int log_d = 0;
    for (int tmp = d; tmp > 1; tmp >>= 1) log_d++;

    auto out = torch::empty_like(x);
    int smem_bytes = d * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "wht_scale_tanh_cuda", [&] {
            wht_scale_tanh_kernel<scalar_t><<<N, d, smem_bytes>>>(
                x.data_ptr<scalar_t>(),
                scale.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                d, log_d, N
            );
        });
    CUDA_CHECK_LAST_ERROR("wht_scale_tanh_cuda kernel launch");

    return out;
}

torch::Tensor parallel_scan_cuda(
    torch::Tensor decay, torch::Tensor x
) {
    // decay: (D,), x: (BD, T)
    int BD = x.size(0);
    int T = x.size(1);
    int D = decay.size(0);
    TORCH_CHECK(T <= MAX_THREADS_PER_BLOCK,
        "parallel_scan_cuda: T=", T, " exceeds max threads per block (",
        MAX_THREADS_PER_BLOCK, "). Use PyTorch fallback for seq_len > 1024.");

    auto y = torch::empty_like(x);
    int smem_bytes = T * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "parallel_scan_cuda", [&] {
            parallel_scan_kernel<scalar_t><<<BD, T, smem_bytes>>>(
                decay.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                BD, T, D
            );
        });
    CUDA_CHECK_LAST_ERROR("parallel_scan_cuda kernel launch");

    return y;
}
"""

_CPP_SRC = r"""
torch::Tensor wht_cuda(torch::Tensor x);
torch::Tensor wht_scale_tanh_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias);
torch::Tensor parallel_scan_cuda(torch::Tensor decay, torch::Tensor x);
"""

# ═══════════════════════════════════════════════════════════════════════
# JIT compilation
# ═══════════════════════════════════════════════════════════════════════

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module

    _klog("JIT compilando kernels CUDA (esto puede tardar ~60s)...")
    t0 = time.time()

    from torch.utils.cpp_extension import load_inline

    build_dir = os.path.join(os.path.dirname(__file__), '..', '.kernel_cache')
    os.makedirs(build_dir, exist_ok=True)
    _klog(f"  build_dir={os.path.abspath(build_dir)}")

    try:
        _module = load_inline(
            name='flux_kernels',
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=['wht_cuda', 'wht_scale_tanh_cuda', 'parallel_scan_cuda'],
            verbose=False,
            extra_cuda_cflags=[
                '-O3', '--use_fast_math',
                '-gencode', 'arch=compute_89,code=sm_89',
                '-gencode', 'arch=compute_89,code=compute_89',
            ],
            build_directory=build_dir,
        )
        dt = time.time() - t0
        _klog(f"  Kernels CUDA compilados OK en {dt:.1f}s")
    except Exception as e:
        dt = time.time() - t0
        _klog(f"  FALLO compilacion kernels tras {dt:.1f}s: {e}")
        raise

    return _module


# ═══════════════════════════════════════════════════════════════════════
# Autograd Functions
# ═══════════════════════════════════════════════════════════════════════

class _WHTFunction(torch.autograd.Function):
    """WHT is its own inverse (orthogonal involution), so backward = forward."""

    @staticmethod
    def forward(ctx, x):
        batch_shape = x.shape[:-1]
        d = x.shape[-1]
        flat = x.reshape(-1, d).contiguous()

        cc = _kernel_call_count
        cc['wht'] += 1
        if cc['wht'] <= _KERNEL_LOG_UNTIL:
            _klog(f"WHT forward #{cc['wht']}: flat.shape={list(flat.shape)}, "
                  f"dtype={flat.dtype}")

        t0 = time.time()
        out = _get_module().wht_cuda(flat)
        if cc['wht'] <= _KERNEL_LOG_UNTIL:
            _klog(f"WHT forward #{cc['wht']}: OK ({time.time()-t0:.4f}s)")

        return out.reshape(*batch_shape, d)

    @staticmethod
    def backward(ctx, grad_output):
        batch_shape = grad_output.shape[:-1]
        d = grad_output.shape[-1]
        flat = grad_output.reshape(-1, d).contiguous()
        out = _get_module().wht_cuda(flat)
        return out.reshape(*batch_shape, d)


class _WHTScaleTanhFunction(torch.autograd.Function):
    """Fused: tanh(scale * WHT(x) + bias)."""

    @staticmethod
    def forward(ctx, x, scale, bias):
        batch_shape = x.shape[:-1]
        d = x.shape[-1]
        flat = x.reshape(-1, d).contiguous()

        cc = _kernel_call_count
        cc['wht_st'] += 1
        log = cc['wht_st'] <= _KERNEL_LOG_UNTIL
        if log:
            _klog(f"WHT_ST forward #{cc['wht_st']}: flat.shape={list(flat.shape)}, "
                  f"scale.shape={list(scale.shape)}, dtype={flat.dtype}")

        t0 = time.time()
        mod = _get_module()
        wht_x = mod.wht_cuda(flat)
        if log:
            _klog(f"WHT_ST #{cc['wht_st']}: wht_cuda OK ({time.time()-t0:.4f}s)")

        t1 = time.time()
        out = torch.tanh(scale * wht_x + bias)
        if log:
            _klog(f"WHT_ST #{cc['wht_st']}: tanh OK ({time.time()-t1:.4f}s), "
                  f"total={time.time()-t0:.4f}s")

        ctx.save_for_backward(wht_x, scale, out)
        ctx.batch_shape = batch_shape
        return out.reshape(*batch_shape, d)

    @staticmethod
    def backward(ctx, grad_output):
        wht_x, scale, tanh_out = ctx.saved_tensors
        batch_shape = ctx.batch_shape
        d = grad_output.shape[-1]
        grad = grad_output.reshape(-1, d)

        dtanh = 1.0 - tanh_out * tanh_out
        common = dtanh * grad
        grad_bias = common.sum(dim=0)
        grad_scale = (common * wht_x).sum(dim=0)

        mod = _get_module()
        grad_x_flat = mod.wht_cuda((scale * common).contiguous())

        return grad_x_flat.reshape(*batch_shape, d), grad_scale, grad_bias


class _ParallelScanFunction(torch.autograd.Function):
    """Fused parallel prefix scan: y[t] = decay * y[t-1] + x[t]."""

    @staticmethod
    def forward(ctx, decay, x):
        B, T, D = x.shape

        cc = _kernel_call_count
        cc['scan'] += 1
        log = cc['scan'] <= _KERNEL_LOG_UNTIL
        if log:
            _klog(f"SCAN forward #{cc['scan']}: B={B} T={T} D={D}, "
                  f"dtype={x.dtype}, decay.shape={list(decay.shape)}")

        t0 = time.time()
        mod = _get_module()

        xt = x.transpose(1, 2).contiguous().reshape(B * D, T)
        if log:
            _klog(f"SCAN #{cc['scan']}: transpose OK, "
                  f"xt.shape={list(xt.shape)}")

        t1 = time.time()
        yt = mod.parallel_scan_cuda(decay, xt)
        if log:
            _klog(f"SCAN #{cc['scan']}: kernel OK ({time.time()-t1:.4f}s)")

        y = yt.reshape(B, D, T).transpose(1, 2).contiguous()
        if log:
            _klog(f"SCAN #{cc['scan']}: reshape OK, total={time.time()-t0:.4f}s")

        ctx.save_for_backward(decay, y)
        ctx.shape = (B, T, D)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        decay, y = ctx.saved_tensors
        B, T, D = ctx.shape
        mod = _get_module()

        grad_flip = grad_output.flip(1).transpose(1, 2).contiguous().reshape(B * D, T)
        grad_scan = mod.parallel_scan_cuda(decay, grad_flip)
        grad_x = grad_scan.reshape(B, D, T).transpose(1, 2).contiguous().flip(1)

        y_prev = F.pad(y[:, :-1, :], (0, 0, 1, 0))
        grad_decay = (grad_x * y_prev).sum(dim=(0, 1))

        return grad_decay, grad_x


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def wht_fused(x: torch.Tensor) -> torch.Tensor:
    """Fused WHT on last dimension (CUDA). 1 kernel launch."""
    return _WHTFunction.apply(x)


def wht_scale_tanh_fused(x: torch.Tensor, scale: torch.Tensor,
                          bias: torch.Tensor) -> torch.Tensor:
    """Fused tanh(scale * WHT(x) + bias). 2 kernel launches."""
    return _WHTScaleTanhFunction.apply(x, scale, bias)


def parallel_scan_fused(decay: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Fused parallel prefix scan.

    Falls back to PyTorch if T > 1024 (CUDA max threads per block).
    """
    T = x.shape[1]
    if T > 1024:
        _klog(f"SCAN fallback: T={T} > 1024, usando PyTorch")
        from .model import parallel_scan
        return parallel_scan(decay, x)
    return _ParallelScanFunction.apply(decay, x)
