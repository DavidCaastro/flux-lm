"""Checkpoint utilities: PyTorch native + Rust V3 binary format interop."""

import struct
import torch
from .model import FluxModel, K


MAGIC = b"ELMC"
VERSION_3 = 3
PRECISION_F32 = 0
PRECISION_F64 = 1


# ── PyTorch native save/load ──────────────────────────────────────────

def save_pytorch(model: FluxModel, optimizer, epoch: int, loss: float,
                 path: str):
    torch.save({
        'd': model.d,
        'n_layers': model.n_layers,
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)


def load_pytorch(path: str, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FluxModel(d=ckpt['d'], n_layers=ckpt['n_layers'])
    model.load_state_dict(ckpt['model_state'])
    return model, ckpt


# ── Rust binary V3 format load ────────────────────────────────────────

def load_rust_checkpoint(path: str, dtype=torch.float32):
    """Load a Rust Flux V3 binary checkpoint into a PyTorch FluxModel."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"

        version = struct.unpack('<I', f.read(4))[0]
        assert version in (2, 3), f"Unsupported version {version}"

        if version == 3:
            precision = struct.unpack('B', f.read(1))[0]
        else:
            precision = PRECISION_F64  # V2 is always f64

        model_type = struct.unpack('<I', f.read(4))[0]
        assert model_type == 4, f"Not a Flux model (type={model_type})"

        d = struct.unpack('<I', f.read(4))[0]
        n_layers = struct.unpack('<I', f.read(4))[0]
        epoch = struct.unpack('<I', f.read(4))[0]
        loss = struct.unpack('<d', f.read(8))[0]
        n_params = struct.unpack('<I', f.read(4))[0]

        if precision == PRECISION_F32:
            fmt_char, fmt_size = 'f', 4
        else:
            fmt_char, fmt_size = 'd', 8

        def read_vec(n):
            raw = f.read(n * fmt_size)
            return list(struct.unpack(f'<{n}{fmt_char}', raw))

        params = read_vec(n_params)
        opt_m = read_vec(n_params)
        opt_v = read_vec(n_params)
        opt_t = struct.unpack('<I', f.read(4))[0]

    # Map flat params to PyTorch model
    model = FluxModel(d=d, n_layers=n_layers)
    _load_flat_params(model, params, d, n_layers)
    model = model.to(dtype)

    return model, {
        'epoch': epoch, 'loss': loss,
        'opt_m': opt_m, 'opt_v': opt_v, 'opt_t': opt_t,
    }


def _load_flat_params(model, params, d, n_layers):
    """Map Rust flat parameter vector to PyTorch module parameters."""
    p = params
    o = 0

    # Embedding: 256*d
    model.embedding.weight.data = torch.tensor(
        p[o:o + 256 * d], dtype=torch.float32,
    ).reshape(256, d)
    o += 256 * d

    # Layers
    layer_size = 525 * d
    for li in range(n_layers):
        layer = model.layers[li]
        lo = o  # layer offset in flat params

        layer.rn_gamma.data = torch.tensor(p[lo:lo + d]).float()
        lo += d

        layer.g_gate.weight.data = torch.tensor(
            p[lo:lo + 256 * d]).float().reshape(256, d)
        lo += 256 * d

        layer.a_bias.weight.data = torch.tensor(
            p[lo:lo + 256 * d]).float().reshape(256, d)
        lo += 256 * d

        layer.w_gate_h.data = torch.tensor(p[lo:lo + d]).float()
        lo += d

        layer.s1.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.b1.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.s2.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.b2.data = torch.tensor(p[lo:lo + d]).float()
        lo += d

        layer.delta_fast.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.b_in_fast.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.c_out_fast.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.skip.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.delta_slow.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.b_in_slow.data = torch.tensor(p[lo:lo + d]).float()
        lo += d
        layer.c_out_slow.data = torch.tensor(p[lo:lo + d]).float()

        o += layer_size

    # Output head
    model.head_w.data = torch.tensor(
        p[o:o + 256 * d]).float().reshape(256, d)
    o += 256 * d
    model.head_b.data = torch.tensor(p[o:o + 256]).float()
    o += 256

    # SPM global
    spm_base = o
    model.spm_w.data = torch.tensor(
        p[spm_base:spm_base + K * d]).float().reshape(K, d)
    spm_o = spm_base + K * d

    for li in range(n_layers):
        layer = model.layers[li]
        layer.spm_delta.data = torch.tensor(
            p[spm_o:spm_o + K]).float()
        spm_o += K
        layer.spm_gate.data = torch.tensor(
            p[spm_o:spm_o + d]).float()
        spm_o += d


def save_rust_checkpoint(model: FluxModel, optimizer_state: dict,
                         epoch: int, loss: float, path: str,
                         precision: str = 'f32'):
    """Save model in Rust-compatible V3 binary format."""
    d = model.d
    n_layers = model.n_layers
    params = _flatten_params(model, d, n_layers)
    n_params = len(params)

    prec_byte = PRECISION_F32 if precision == 'f32' else PRECISION_F64
    fmt_char = 'f' if precision == 'f32' else 'd'

    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION_3))
        f.write(struct.pack('B', prec_byte))
        f.write(struct.pack('<I', 4))  # model_type = Flux
        f.write(struct.pack('<I', d))
        f.write(struct.pack('<I', n_layers))
        f.write(struct.pack('<I', epoch))
        f.write(struct.pack('<d', loss))
        f.write(struct.pack('<I', n_params))

        def write_vec(vec):
            f.write(struct.pack(f'<{len(vec)}{fmt_char}', *vec))

        write_vec(params)
        # Write optimizer m and v as zeros (or from state)
        zeros = [0.0] * n_params
        write_vec(zeros)
        write_vec(zeros)
        f.write(struct.pack('<I', 0))


def _flatten_params(model, d, n_layers):
    """Flatten PyTorch model params into Rust layout."""
    params = []

    # Embedding
    params.extend(model.embedding.weight.data.cpu().float().reshape(-1).tolist())

    # Layers
    for li in range(n_layers):
        layer = model.layers[li]
        params.extend(layer.rn_gamma.data.cpu().float().tolist())
        params.extend(layer.g_gate.weight.data.cpu().float().reshape(-1).tolist())
        params.extend(layer.a_bias.weight.data.cpu().float().reshape(-1).tolist())
        params.extend(layer.w_gate_h.data.cpu().float().tolist())
        params.extend(layer.s1.data.cpu().float().tolist())
        params.extend(layer.b1.data.cpu().float().tolist())
        params.extend(layer.s2.data.cpu().float().tolist())
        params.extend(layer.b2.data.cpu().float().tolist())
        params.extend(layer.delta_fast.data.cpu().float().tolist())
        params.extend(layer.b_in_fast.data.cpu().float().tolist())
        params.extend(layer.c_out_fast.data.cpu().float().tolist())
        params.extend(layer.skip.data.cpu().float().tolist())
        params.extend(layer.delta_slow.data.cpu().float().tolist())
        params.extend(layer.b_in_slow.data.cpu().float().tolist())
        params.extend(layer.c_out_slow.data.cpu().float().tolist())

    # Output head
    params.extend(model.head_w.data.cpu().float().reshape(-1).tolist())
    params.extend(model.head_b.data.cpu().float().tolist())

    # SPM global
    params.extend(model.spm_w.data.cpu().float().reshape(-1).tolist())
    for li in range(n_layers):
        layer = model.layers[li]
        params.extend(layer.spm_delta.data.cpu().float().tolist())
        params.extend(layer.spm_gate.data.cpu().float().tolist())

    return params
