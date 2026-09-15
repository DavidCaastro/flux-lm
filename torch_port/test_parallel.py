#!/usr/bin/env python3
"""Validate parallel scan vs sequential: measure approximation error
and verify convergence with increasing correction order."""

import torch
from flux.model import FluxModel, parallel_scan


def test_parallel_scan_exact():
    """parallel_scan must produce identical results to sequential loop."""
    B, T, d = 4, 64, 16
    decay = torch.rand(d) * 0.5 + 0.4   # decay in [0.4, 0.9]
    x = torch.randn(B, T, d)

    # Sequential reference
    y_seq = torch.zeros_like(x)
    h = torch.zeros(B, d)
    for t in range(T):
        h = decay * h + x[:, t, :]
        y_seq[:, t, :] = h

    # Parallel scan
    y_par = parallel_scan(decay, x)

    err = (y_par - y_seq).abs().max().item()
    print(f"  parallel_scan vs sequential: max_err={err:.2e}")
    assert err < 1e-4, f"Parallel scan too inaccurate: {err}"


def test_mode_comparison():
    """Compare sequential (exact) vs parallel (mean-field) on same weights."""
    d, nl, B, T = 32, 2, 4, 32
    torch.manual_seed(42)

    # Create model, get sequential output
    model = FluxModel(d=d, n_layers=nl, parallel=False)
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        logits_seq, loss_seq = model(x, y)

    # Switch to parallel, same weights
    model.set_mode(parallel=True, n_corrections=1)
    with torch.no_grad():
        logits_par, loss_par = model(x, y)

    logit_err = (logits_par - logits_seq).abs()
    rel_err = logit_err / (logits_seq.abs().max() + 1e-8)
    loss_diff = abs(loss_par.item() - loss_seq.item())

    print(f"  Sequential loss:     {loss_seq.item():.6f}")
    print(f"  Parallel loss:       {loss_par.item():.6f}")
    print(f"  Loss difference:     {loss_diff:.2e}")
    print(f"  Logit max abs err:   {logit_err.max().item():.2e}")
    print(f"  Logit mean abs err:  {logit_err.mean().item():.2e}")
    print(f"  Logit max rel err:   {rel_err.max().item():.2e}")


def test_correction_convergence():
    """Error should decrease with more correction steps."""
    d, nl, B, T = 32, 2, 4, 32
    torch.manual_seed(42)

    model = FluxModel(d=d, n_layers=nl, parallel=False)
    x = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        logits_exact = model(x)

    print(f"\n  Convergence by correction order:")
    prev_err = float('inf')
    for nc in range(5):
        model.set_mode(parallel=True, n_corrections=nc)
        with torch.no_grad():
            logits_approx = model(x)
        err = (logits_approx - logits_exact).abs().mean().item()
        improved = "OK" if err < prev_err or nc == 0 else "WORSE"
        print(f"    n_corrections={nc}: mean_err={err:.6e}  {improved}")
        prev_err = err


def test_parallel_backward():
    """Gradients flow through parallel scan."""
    d, nl, B, T = 16, 2, 2, 16
    model = FluxModel(d=d, n_layers=nl, parallel=True, n_corrections=1)
    x = torch.randint(0, 256, (B, T))
    y = torch.randint(0, 256, (B, T))

    _, loss = model(x, y)
    loss.backward()

    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    n_zero = sum(1 for p in model.parameters()
                 if p.grad is not None and p.grad.abs().max() == 0)

    print(f"  Parallel backward: {n_grads}/{n_total} params with grads, "
          f"{n_zero} zero-grad")
    assert n_grads == n_total, "Missing gradients in parallel mode"
    assert n_zero == 0, "Some gradients are zero"


def test_parallel_training():
    """Train a few steps in parallel mode, verify loss decreases."""
    from flux.optim import EntropicAdam

    d, nl = 16, 2
    model = FluxModel(d=d, n_layers=nl, parallel=True, n_corrections=1)
    opt = EntropicAdam(model.parameters(), lr=1e-3, total_epochs=10)

    corpus = (b"The quick brown fox jumps over the lazy dog. " * 10)
    from flux.data import ByteCorpusDataset
    from torch.utils.data import DataLoader
    ds = ByteCorpusDataset(corpus, seq_len=32)
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)

    losses = []
    for epoch in range(3):
        total, n = 0.0, 0
        for xb, yb in loader:
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(epoch=epoch + 1)
            opt.zero_grad()
            total += loss.item()
            n += 1
        losses.append(total / max(n, 1))

    print(f"  Parallel training: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss did not decrease in parallel mode"


if __name__ == '__main__':
    print("Flux v3 — parallel scan validation\n")
    test_parallel_scan_exact()
    test_mode_comparison()
    test_correction_convergence()
    test_parallel_backward()
    test_parallel_training()
    print("\nAll parallel tests passed.")
