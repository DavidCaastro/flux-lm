#!/usr/bin/env python3
"""Quick smoke test: train 3 epochs on tiny synthetic corpus, verify loss decreases."""

import torch
from flux.model import FluxModel
from flux.optim import EntropicAdam
from flux.data import ByteCorpusDataset
from torch.utils.data import DataLoader

def main():
    d, nl = 16, 2
    seq_len, batch_size, epochs = 32, 4, 3

    # Tiny corpus: repeated pattern so model can learn it
    corpus = (b"Hello world! This is Flux v3. " * 20)
    ds = ByteCorpusDataset(corpus, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = FluxModel(d=d, n_layers=nl)
    opt = EntropicAdam(model.parameters(), lr=1e-3, total_epochs=epochs)

    losses = []
    for epoch in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for x, y in loader:
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(epoch=epoch)
            opt.zero_grad()
            total_loss += loss.item()
            n += 1
        avg = total_loss / max(n, 1)
        losses.append(avg)
        print(f"  epoch {epoch}: loss={avg:.4f} bpb={avg / 0.693:.3f}")

    assert losses[-1] < losses[0], \
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"\n  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f} OK")

    # Test generation
    from flux.generate import generate
    out = generate(model, b"Hello", length=50, temperature=0.8, device='cpu')
    print(f"  Generated {len(out)} bytes: {out[:40]}...")
    print("\nSmoke test passed.")


if __name__ == '__main__':
    main()
