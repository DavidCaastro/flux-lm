#!/usr/bin/env python3
"""CLI for Flux v3 text generation."""

import argparse
import torch
from flux.checkpoint import load_pytorch, load_rust_checkpoint
from flux.generate import generate


def main():
    p = argparse.ArgumentParser(description='Flux v3 Generation')
    p.add_argument('--ckpt', type=str, help='PyTorch checkpoint (.pt)')
    p.add_argument('--rust-ckpt', type=str, help='Rust checkpoint (.bin)')
    p.add_argument('--seed-text', type=str, default='ROMEO:')
    p.add_argument('--length', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    if not args.ckpt and not args.rust_ckpt:
        print('ERROR: --ckpt or --rust-ckpt required')
        return

    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.rust_ckpt:
        model, info = load_rust_checkpoint(args.rust_ckpt)
    else:
        model, info = load_pytorch(args.ckpt, device='cpu')

    model = model.to(device).eval()
    print(f'Loaded: d={model.d}, layers={model.n_layers}, '
          f'params={model.count_params():,}')

    seed_bytes = args.seed_text.encode('utf-8')
    output = generate(model, seed_bytes, args.length,
                      temperature=args.temperature, device=device)

    print(args.seed_text, end='')
    print(output.decode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
