#!/usr/bin/env python3
"""Quick inference test for Flux LM checkpoint."""
import torch
import torch.nn.functional as F
from flux.checkpoint import load_pytorch

def generate(model, prompt, max_new=150, temperature=0.8):
    byte_ids = list(prompt.encode("utf-8"))
    context = byte_ids[:]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([context[-256:]], dtype=torch.long)
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            context.append(next_byte)
            if len(context) > len(byte_ids) + 5 and context[-2:] == [10, 10]:
                break
    return bytes(context[len(byte_ids):]).decode("utf-8", errors="replace")

def main():
    model, info = load_pytorch("checkpoints/flux_epoch_0050.pt", device="cpu")
    model.eval()
    model.set_mode(parallel=False)
    print(f"Loaded epoch {info['epoch']}, params={model.count_params():,}")
    print()

    prompts = [
        "def fibonacci(n):",
        "class DataLoader:",
        "import os\nimport sys\n\ndef main():",
        "for i in range(10):",
        "def parse_args():",
    ]

    for p in prompts:
        print(f"PROMPT: {p}")
        out = generate(model, p)
        print(f"OUTPUT:{out}")
        print("-" * 60)

if __name__ == "__main__":
    main()
