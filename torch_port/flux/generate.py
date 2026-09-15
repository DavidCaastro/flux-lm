"""Text generation for Flux v3."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, seed_bytes: bytes, length: int,
             temperature: float = 0.8, device: str = 'cuda',
             max_ctx: int = 256) -> bytes:
    """Autoregressive byte-level generation."""
    model.eval()
    seq = list(seed_bytes)

    for _ in range(length):
        start = max(0, len(seq) - max_ctx)
        ctx = torch.tensor(seq[start:], dtype=torch.long,
                           device=device).unsqueeze(0)   # (1, T)

        logits = model(ctx)                                # (1, T, 256)
        logits = logits[0, -1, :]                          # (256,)

        if abs(temperature - 1.0) > 1e-6:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        chosen = torch.multinomial(probs, 1).item()
        seq.append(chosen)

    return bytes(seq[len(seed_bytes):])
