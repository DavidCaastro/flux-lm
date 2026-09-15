"""Byte-level corpus dataset for Flux v3 training."""

import torch
from torch.utils.data import Dataset


class ByteCorpusDataset(Dataset):
    """Dataset that yields (input, target) pairs of byte sequences.

    Each sample is a contiguous chunk of seq_len bytes from the corpus.
    Target is the next-byte prediction (shifted by 1).
    """

    def __init__(self, data: bytes | bytearray, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        self.n_chunks = (len(data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        t = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()
        return t[:-1], t[1:]


def load_corpus(path: str, train_frac: float = 0.9):
    """Load corpus and split into train/test byte arrays."""
    with open(path, 'rb') as f:
        data = f.read()
    split = int(len(data) * train_frac)
    split = max(split, 2)
    split = min(split, len(data) - 2)
    return data[:split], data[split:]
