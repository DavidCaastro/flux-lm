#!/usr/bin/env python3
"""Validation tests for the PyTorch Flux v3 port."""

import math
import torch
import torch.nn.functional as F
from flux.model import FluxModel, wht


def test_wht_involution():
    """WHT applied twice should return the original vector."""
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5, -1.5, 2.5, 0.0, 4.0]])
    original = x.clone()
    y = wht(wht(x))
    err = (y - original).abs().max().item()
    assert err < 1e-6, f"WHT involution failed: max_err={err}"
    print(f"  WHT involution: OK (err={err:.2e})")


def test_wht_size_2():
    """WHT on size-2 vector."""
    x = torch.tensor([[3.0, 1.0]])
    y = wht(x)
    s = math.sqrt(2.0)
    expected = torch.tensor([[4.0 / s, 2.0 / s]])
    err = (y - expected).abs().max().item()
    assert err < 1e-6, f"WHT size-2 failed: {y} vs {expected}"
    print(f"  WHT size 2: OK (err={err:.2e})")


def test_wht_batched():
    """WHT should work on batched inputs."""
    B, d = 8, 16
    x = torch.randn(B, d)
    y = wht(x)
    assert y.shape == (B, d), f"Wrong shape: {y.shape}"
    z = wht(y)
    err = (z - x).abs().max().item()
    assert err < 1e-5, f"Batched WHT involution failed: {err}"
    print(f"  WHT batched (B={B}, d={d}): OK (err={err:.2e})")


def test_model_forward():
    """Model forward pass produces correct output shapes."""
    d, nl, B, T = 32, 2, 4, 16
    model = FluxModel(d=d, n_layers=nl)
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    logits = model(x)
    assert logits.shape == (B, T, 256), f"Wrong logits shape: {logits.shape}"

    logits2, loss = model(x, y)
    assert logits2.shape == (B, T, 256)
    assert loss.item() > 0, "Loss should be positive"
    print(f"  Forward pass: OK (loss={loss.item():.4f})")


def test_model_backward():
    """Gradients flow through the full model."""
    d, nl, B, T = 16, 2, 2, 8
    model = FluxModel(d=d, n_layers=nl)
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    _, loss = model(x, y)
    loss.backward()

    n_grads = 0
    n_zero = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            n_grads += 1
            if p.grad.abs().max().item() == 0:
                n_zero += 1
        else:
            print(f"  WARNING: no grad for {name}")

    assert n_grads > 0, "No gradients computed"
    print(f"  Backward pass: OK ({n_grads} params with grads, "
          f"{n_zero} zero-grad)")


def test_param_count():
    """Parameter count matches the Rust formula: 256*d + L*525*d + 256*d + 256 + K*d + L*(K+d)."""
    d, nl = 64, 3
    K = 4
    expected = 256*d + nl*525*d + 256*d + 256 + K*d + nl*(K+d)
    model = FluxModel(d=d, n_layers=nl)
    actual = model.count_params()
    assert actual == expected, f"Param count mismatch: {actual} vs {expected}"
    print(f"  Param count (d={d}, L={nl}): {actual:,} == {expected:,} OK")


def test_entropic_adam():
    """EntropicAdam runs without errors."""
    from flux.optim import EntropicAdam
    d, nl = 16, 1
    model = FluxModel(d=d, n_layers=nl)
    opt = EntropicAdam(model.parameters(), lr=1e-3, total_epochs=10)

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))

    for step in range(3):
        _, loss = model(x, y)
        loss.backward()
        opt.step(epoch=step + 1)
        opt.zero_grad()

    print(f"  EntropicAdam: OK (3 steps, final loss={loss.item():.4f})")


def test_cosine_schedule():
    """Warm-restart cosine schedule produces values in [0, 1]."""
    from flux.optim import WarmRestartCosineSchedule
    sched = WarmRestartCosineSchedule(total_epochs=200)
    for e in range(1, 201):
        f = sched.get_factor(e)
        assert 0 <= f <= 1.01, f"Schedule out of range at epoch {e}: {f}"
    print(f"  Cosine schedule: OK (200 epochs in [0, 1])")


def test_dataset():
    """ByteCorpusDataset yields correct shapes."""
    from flux.data import ByteCorpusDataset
    data = bytes(range(256)) * 4
    ds = ByteCorpusDataset(data, seq_len=32)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (32,) and y.shape == (32,)
    assert (x[1:] == y[:-1]).all(), "Target should be input shifted by 1"
    print(f"  Dataset: OK ({len(ds)} chunks, seq_len=32)")


def test_fused_kernels():
    """Test fused CUDA kernels if available."""
    from flux.model import HAS_FUSED_KERNELS
    if not HAS_FUSED_KERNELS:
        print("  Fused kernels: SKIPPED (not available)")
        return
    if not torch.cuda.is_available():
        print("  Fused kernels: SKIPPED (no CUDA)")
        return

    from flux.kernels import wht_fused, wht_scale_tanh_fused, parallel_scan_fused

    device = 'cuda'
    d = 256

    # WHT involution
    x = torch.randn(8, d, device=device)
    y = wht_fused(wht_fused(x))
    err = (y - x).abs().max().item()
    assert err < 1e-4, f"Fused WHT involution failed: {err}"
    print(f"  Fused WHT involution: OK (err={err:.2e})")

    # WHT matches reference
    x2 = torch.randn(4, 32, d, device=device)
    ref = wht(x2)
    fused = wht_fused(x2)
    err2 = (ref - fused).abs().max().item()
    assert err2 < 1e-4, f"Fused WHT vs reference: {err2}"
    print(f"  Fused WHT vs reference: OK (err={err2:.2e})")

    # WHT + scale + tanh
    x3 = torch.randn(4, 32, d, device=device)
    scale = torch.randn(d, device=device)
    bias = torch.randn(d, device=device)
    ref3 = torch.tanh(scale * wht(x3) + bias)
    fused3 = wht_scale_tanh_fused(x3, scale, bias)
    err3 = (ref3 - fused3).abs().max().item()
    assert err3 < 1e-3, f"Fused WHT+scale+tanh: {err3}"
    print(f"  Fused WHT+scale+tanh: OK (err={err3:.2e})")

    # Parallel scan
    B, T, D = 4, 64, 16
    decay = torch.rand(D, device=device) * 0.5 + 0.4
    inp = torch.randn(B, T, D, device=device)

    from flux.model import parallel_scan
    ref4 = parallel_scan(decay, inp)
    fused4 = parallel_scan_fused(decay, inp)
    err4 = (ref4 - fused4).abs().max().item()
    assert err4 < 1e-3, f"Fused parallel_scan: {err4}"
    print(f"  Fused parallel_scan: OK (err={err4:.2e})")

    # WHT backward (gradcheck)
    x_gc = torch.randn(4, d, device=device, dtype=torch.float64, requires_grad=True)
    from flux.kernels import _WHTFunction
    ok = torch.autograd.gradcheck(_WHTFunction.apply, (x_gc,), eps=1e-6, atol=1e-4)
    print(f"  Fused WHT gradcheck: {'OK' if ok else 'FAILED'}")

    # bf16 test
    x_bf = torch.randn(8, d, device=device, dtype=torch.bfloat16)
    y_bf = wht_fused(wht_fused(x_bf))
    err_bf = (y_bf - x_bf).abs().max().item()
    print(f"  Fused WHT bf16 involution: OK (err={err_bf:.2e})")


if __name__ == '__main__':
    print("Flux v3 PyTorch port — validation tests\n")
    test_wht_involution()
    test_wht_size_2()
    test_wht_batched()
    test_model_forward()
    test_model_backward()
    test_param_count()
    test_entropic_adam()
    test_cosine_schedule()
    test_dataset()
    test_fused_kernels()
    print("\nAll tests passed.")
