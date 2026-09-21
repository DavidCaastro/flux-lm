#!/usr/bin/env python3
"""Comparative evaluation: 3.5M vs 7M Flux models.

Evaluates both models on:
1. In-distribution: test split of corpus_python.txt (same split as training)
2. Out-of-distribution: novel Python code NOT in training corpus

Metrics per sample:
- BPB (bits per byte)
- Top-1 byte prediction accuracy
- Loss (nats)

Runs on CPU. No GPU required.

Usage:
    python eval_comparison.py
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux.model import FluxModel
from flux.checkpoint import load_pytorch
import importlib.util

LN2 = math.log(2)

# ── Paths ─────────────────────────────────────────────────────────────
CKPT_35M = os.path.join(os.path.dirname(__file__), "checkpoints", "flux_epoch_0050.pt")
CKPT_7M = os.path.join(os.path.dirname(__file__), "checkpoints_7m_v3", "flux_epoch_0067.pt")
CORPUS = os.path.join(os.path.dirname(__file__), "data", "corpus_python.txt")


# ═══════════════════════════════════════════════════════════════════════
# Out-of-distribution samples — code NOT from corpus repos
# Corpus repos: django, flask, fastapi, scikit-learn, pandas, numpy,
# transformers, requests, pydantic, click, poetry, pip, celery,
# matplotlib, sympy, sqlalchemy, scrapy, ansible, pytest, etc.
#
# OOD sources: algorithms from scratch, stdlib patterns, game logic,
# cryptography, modern Python 3.10+, functional style, finance, etc.
# ═══════════════════════════════════════════════════════════════════════

OOD_SAMPLES = {
    "algorithm_quicksort": '''\
def quicksort(arr):
    """In-place quicksort with Lomuto partition scheme."""
    def partition(lo, hi):
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        return i + 1

    def sort(lo, hi):
        if lo < hi:
            p = partition(lo, hi)
            sort(lo, p - 1)
            sort(p + 1, hi)

    if len(arr) > 1:
        sort(0, len(arr) - 1)
    return arr


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',

    "datastructure_linked_list": '''\
class Node:
    __slots__ = ('value', 'next')

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def push_front(self, value):
        self.head = Node(value, self.head)
        self.size += 1

    def pop_front(self):
        if self.head is None:
            raise IndexError("pop from empty list")
        value = self.head.value
        self.head = self.head.next
        self.size -= 1
        return value

    def find(self, value):
        current = self.head
        while current is not None:
            if current.value == value:
                return True
            current = current.next
        return False

    def reverse(self):
        prev = None
        current = self.head
        while current is not None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def __len__(self):
        return self.size

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self):
        items = " -> ".join(str(x) for x in self)
        return f"LinkedList([{items}])"
''',

    "modern_python_match": '''\
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable


class TokenType(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    LPAREN = auto()
    RPAREN = auto()
    PLUS = auto()
    MINUS = auto()
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    line: int
    col: int


@dataclass
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass
class Literal:
    value: int | float | str


@dataclass
class Ident:
    name: str


Expr = BinOp | Literal | Ident


def eval_expr(expr: Expr) -> int | float:
    match expr:
        case Literal(value=v):
            return v
        case BinOp(op="+", left=l, right=r):
            return eval_expr(l) + eval_expr(r)
        case BinOp(op="-", left=l, right=r):
            return eval_expr(l) - eval_expr(r)
        case BinOp(op="*", left=l, right=r):
            return eval_expr(l) * eval_expr(r)
        case _:
            raise ValueError(f"Unknown expression: {expr}")
''',

    "graph_algorithms": '''\
from collections import defaultdict, deque
import heapq


class Graph:
    def __init__(self, directed=False):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def bfs(self, start):
        visited = {start}
        queue = deque([start])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def dijkstra(self, start):
        dist = {start: 0}
        heap = [(0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float('inf')):
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def topological_sort(self):
        in_degree = defaultdict(int)
        for u in self.adj:
            for v, _ in self.adj[u]:
                in_degree[v] += 1
        queue = deque(u for u in self.adj if in_degree[u] == 0)
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in self.adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return order
''',

    "crypto_hash": '''\
import struct


def sha256_pad(message: bytes) -> bytes:
    length = len(message)
    bit_length = length * 8
    message += b"\\x80"
    while (len(message) + 8) % 64 != 0:
        message += b"\\x00"
    message += struct.pack(">Q", bit_length)
    return message


def right_rotate(value, amount):
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF


K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
]


def sha256_compress(block: bytes, h: list) -> list:
    assert len(block) == 64
    w = list(struct.unpack(">16L", block))
    for i in range(16, 64):
        s0 = right_rotate(w[i-15], 7) ^ right_rotate(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = right_rotate(w[i-2], 17) ^ right_rotate(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF)

    a, b, c, d, e, f, g, hh = h
    for i in range(64):
        s1 = right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)
        ch = (e & f) ^ (~e & g)
        temp1 = (hh + s1 + ch + K[i % 16] + w[i]) & 0xFFFFFFFF
        s0 = right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + maj) & 0xFFFFFFFF
        hh, g, f, e, d, c, b, a = g, f, e, (d + temp1) & 0xFFFFFFFF, c, b, a, (temp1 + temp2) & 0xFFFFFFFF

    return [(x + y) & 0xFFFFFFFF for x, y in zip(h, [a, b, c, d, e, f, g, hh])]
''',

    "functional_pipeline": '''\
from functools import reduce, partial
from itertools import chain, islice, groupby
from operator import itemgetter


def pipe(value, *functions):
    return reduce(lambda v, f: f(v), functions, value)


def compose(*functions):
    def composed(value):
        for f in reversed(functions):
            value = f(value)
        return value
    return composed


def chunk(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def flatten(nested):
    for item in nested:
        if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def group_by_key(records, key):
    sorted_records = sorted(records, key=itemgetter(key))
    return {
        k: list(v)
        for k, v in groupby(sorted_records, key=itemgetter(key))
    }


def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    wrapper.cache_clear = cache.clear
    return wrapper


@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',

    "socket_server": '''\
import socket
import threading
import logging

logger = logging.getLogger(__name__)


class TCPServer:
    def __init__(self, host="0.0.0.0", port=8080, max_connections=128):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.server_socket = None
        self.running = False
        self.threads = []

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.max_connections)
        self.running = True
        logger.info(f"Server listening on {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"Connection from {address}")
                thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True,
                )
                thread.start()
                self.threads.append(thread)
            except OSError:
                break

    def handle_client(self, client_socket, address):
        try:
            buffer = b""
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                buffer += data
                while b"\\n" in buffer:
                    line, buffer = buffer.split(b"\\n", 1)
                    response = self.process_message(line.decode("utf-8"))
                    client_socket.sendall(response.encode("utf-8") + b"\\n")
        except Exception as e:
            logger.error(f"Error with {address}: {e}")
        finally:
            client_socket.close()
            logger.info(f"Disconnected: {address}")

    def process_message(self, message):
        return f"ECHO: {message}"

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
''',

    "numerical_methods": '''\
import math


def newton_raphson(f, df, x0, tol=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            raise ValueError("Derivative too close to zero")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    raise RuntimeError(f"Did not converge in {max_iter} iterations")


def trapezoidal_rule(f, a, b, n=1000):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def runge_kutta_4(f, y0, t0, t_end, dt):
    t = t0
    y = y0
    trajectory = [(t, y)]
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt * k1 / 2)
        k3 = f(t + dt / 2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)
        y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        trajectory.append((t, y))
    return trajectory


def lu_decomposition(matrix):
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            total = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = matrix[i][k] - total
        for k in range(i, n):
            if i == k:
                L[i][i] = 1.0
            else:
                total = sum(L[k][j] * U[j][i] for j in range(i))
                L[k][i] = (matrix[k][i] - total) / U[i][i]
    return L, U
''',

    "game_logic_ecs": '''\
from dataclasses import dataclass
from typing import Any


@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Velocity:
    dx: float = 0.0
    dy: float = 0.0


@dataclass
class Health:
    current: int = 100
    maximum: int = 100

    @property
    def ratio(self):
        return self.current / self.maximum

    def damage(self, amount):
        self.current = max(0, self.current - amount)

    def heal(self, amount):
        self.current = min(self.maximum, self.current + amount)

    @property
    def alive(self):
        return self.current > 0


class World:
    def __init__(self):
        self.entities = {}
        self.components = {}
        self.next_id = 0

    def spawn(self, **components):
        eid = self.next_id
        self.next_id += 1
        self.entities[eid] = set()
        for comp_type, comp in components.items():
            if comp_type not in self.components:
                self.components[comp_type] = {}
            self.components[comp_type][eid] = comp
            self.entities[eid].add(comp_type)
        return eid

    def get(self, eid, comp_type):
        return self.components.get(comp_type, {}).get(eid)

    def query(self, *comp_types):
        result = []
        for eid in self.entities:
            if all(ct in self.entities[eid] for ct in comp_types):
                comps = tuple(self.components[ct][eid] for ct in comp_types)
                result.append((eid, *comps))
        return result

    def destroy(self, eid):
        if eid in self.entities:
            for comp_type in self.entities[eid]:
                del self.components[comp_type][eid]
            del self.entities[eid]


def movement_system(world, dt):
    for eid, pos, vel in world.query("position", "velocity"):
        pos.x += vel.dx * dt
        pos.y += vel.dy * dt


def damage_system(world, attacker_id, target_id, amount):
    health = world.get(target_id, "health")
    if health and health.alive:
        health.damage(amount)
        if not health.alive:
            world.destroy(target_id)
''',

    "cli_tool_argparse": '''\
import argparse
import csv
import sys
from pathlib import Path
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze CSV files and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input CSV file")
    parser.add_argument("-o", "--output", type=Path, help="Output file")
    parser.add_argument("-c", "--column", required=True, help="Column to analyze")
    parser.add_argument("--top", type=int, default=10, help="Show top N values")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def read_csv(path, delimiter, has_header):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader)
        else:
            header = None
        rows = list(reader)
    return header, rows


def analyze_column(rows, col_idx):
    values = [row[col_idx] for row in rows if col_idx < len(row)]
    counter = Counter(values)
    total = len(values)
    unique = len(counter)
    most_common = counter.most_common()
    return {
        "total": total,
        "unique": unique,
        "most_common": most_common,
        "empty": sum(1 for v in values if not v.strip()),
    }


def format_report(stats, column_name, top_n):
    lines = [
        f"Column: {column_name}",
        f"Total values: {stats['total']}",
        f"Unique values: {stats['unique']}",
        f"Empty values: {stats['empty']}",
        f"",
        f"Top {top_n} values:",
    ]
    for value, count in stats["most_common"][:top_n]:
        pct = 100 * count / stats["total"]
        lines.append(f"  {value!r:30s} {count:6d} ({pct:5.1f}%)")
    return "\\n".join(lines)


def main():
    args = parse_args()
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    header, rows = read_csv(args.input, args.delimiter, not args.no_header)
    if header and args.column in header:
        col_idx = header.index(args.column)
    else:
        col_idx = int(args.column)
    stats = analyze_column(rows, col_idx)
    report = format_report(stats, args.column, args.top)
    if args.output:
        args.output.write_text(report)
    else:
        print(report)
''',
}


# ═══════════════════════════════════════════════════════════════════════
# Evaluation functions
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_sample(model, text_bytes, seq_len=256):
    """Compute BPB, loss, and top-1 accuracy for a byte sequence."""
    if len(text_bytes) < 2:
        return None

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_bytes = 0

    for i in range(0, len(text_bytes) - 1, seq_len):
        chunk = text_bytes[i:i + seq_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long)
        y = torch.tensor([chunk[1:]], dtype=torch.long)

        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1), reduction='sum')
        preds = logits.argmax(dim=-1)
        correct = (preds.view(-1) == y.view(-1)).sum().item()

        n = len(chunk) - 1
        total_loss += loss.item()
        total_correct += correct
        total_bytes += n

    if total_bytes == 0:
        return None

    avg_loss = total_loss / total_bytes
    bpb = avg_loss / LN2
    accuracy = total_correct / total_bytes
    return {
        "bpb": bpb,
        "loss_nats": avg_loss,
        "accuracy_top1": accuracy,
        "total_bytes": total_bytes,
    }


def evaluate_corpus_split(model, data_bytes, seq_len=256, max_samples=20,
                          sample_size=2048, label=""):
    """Evaluate model on random chunks of a byte corpus."""
    import random
    random.seed(42)

    results = []
    max_start = len(data_bytes) - sample_size - 1
    if max_start < 0:
        max_start = 0
        sample_size = len(data_bytes) - 1

    offsets = sorted(random.sample(range(max_start), min(max_samples, max_start)))

    for idx, offset in enumerate(offsets):
        chunk = list(data_bytes[offset:offset + sample_size])
        t0 = time.time()
        result = evaluate_sample(model, chunk, seq_len=seq_len)
        dt = time.time() - t0
        if result:
            result["offset"] = offset
            result["time_s"] = dt
            results.append(result)
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"  [{label}] {idx+1}/{len(offsets)}: "
                      f"BPB={result['bpb']:.3f}, acc={result['accuracy_top1']:.3f}, "
                      f"{dt:.1f}s")

    return results


def evaluate_ood_samples(model, samples_dict, seq_len=256, label=""):
    """Evaluate model on named OOD code samples."""
    results = {}
    for name, code in samples_dict.items():
        data = list(code.encode('utf-8'))
        t0 = time.time()
        result = evaluate_sample(model, data, seq_len=seq_len)
        dt = time.time() - t0
        if result:
            result["name"] = name
            result["time_s"] = dt
            results[name] = result
            print(f"  [{label}] {name}: "
                  f"BPB={result['bpb']:.3f}, acc={result['accuracy_top1']:.3f}, "
                  f"{len(data)} bytes, {dt:.1f}s")
    return results


def aggregate_results(results_list):
    """Compute aggregate stats from a list of result dicts."""
    if not results_list:
        return {}
    bpbs = [r["bpb"] for r in results_list]
    accs = [r["accuracy_top1"] for r in results_list]
    total_bytes = sum(r["total_bytes"] for r in results_list)
    total_loss = sum(r["loss_nats"] * r["total_bytes"] for r in results_list)
    weighted_bpb = (total_loss / total_bytes) / LN2 if total_bytes > 0 else 0

    return {
        "weighted_bpb": weighted_bpb,
        "mean_bpb": sum(bpbs) / len(bpbs),
        "min_bpb": min(bpbs),
        "max_bpb": max(bpbs),
        "std_bpb": (sum((b - sum(bpbs)/len(bpbs))**2 for b in bpbs) / len(bpbs)) ** 0.5,
        "mean_accuracy": sum(accs) / len(accs),
        "min_accuracy": min(accs),
        "max_accuracy": max(accs),
        "total_bytes": total_bytes,
        "n_samples": len(results_list),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("FLUX LM — Evaluacion comparativa 3.5M vs 7M")
    print("=" * 70)

    # ── Load corpus ──
    print("\n[1/5] Cargando corpus...")
    with open(CORPUS, "rb") as f:
        raw = f.read()
    split = int(len(raw) * 0.9)
    train_data = raw[:split]
    test_data = raw[split:]
    print(f"  Corpus: {len(raw):,} bytes")
    print(f"  Train: {len(train_data):,} bytes, Test: {len(test_data):,} bytes")

    # ── Load models ──
    # 3.5M: use OLD model code (pre-SPM-enhancements, commit 4c20b52)
    # The 3.5M checkpoint is incompatible with current code due to:
    #   - E2: spm_w_dec (asymmetric projections) didn't exist
    #   - E3: output was additive (base + cond), not multiplicative ((1+cond)*base)
    #   - E1: spm_delta_mod didn't exist
    #   - w_c_fast/w_c_slow (dynamic c_out gating) didn't exist
    print("\n[2/5] Cargando modelos...")
    t0 = time.time()
    old_model_path = os.path.join(os.path.dirname(__file__), "flux", "model_old.py")
    spec = importlib.util.spec_from_file_location("model_old", old_model_path)
    model_old_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_old_mod)
    ckpt_35m = torch.load(CKPT_35M, map_location='cpu', weights_only=False)
    model_35m = model_old_mod.FluxModel(
        d=ckpt_35m['d'], n_layers=ckpt_35m['n_layers'], parallel=False)
    model_35m.load_state_dict(ckpt_35m['model_state'], strict=False)
    model_35m.eval()
    print(f"  3.5M cargado (old arch): d={model_35m.d}, params={model_35m.count_params():,} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    model_7m, _ = load_pytorch(CKPT_7M, device='cpu')
    model_7m.eval()
    model_7m.set_mode(parallel=False)
    print(f"  7M cargado: d={model_7m.d}, params={model_7m.count_params():,} ({time.time()-t0:.1f}s)")

    # ── In-distribution: test split ──
    print("\n[3/5] Evaluacion IN-DISTRIBUTION (test split del corpus)...")
    print(f"  20 muestras de 2048 bytes cada una")

    print("\n  --- 3.5M ---")
    indist_35m = evaluate_corpus_split(
        model_35m, test_data, max_samples=20, sample_size=2048, label="3.5M-ID")
    agg_35m_id = aggregate_results(indist_35m)

    print("\n  --- 7M ---")
    indist_7m = evaluate_corpus_split(
        model_7m, test_data, max_samples=20, sample_size=2048, label="7M-ID")
    agg_7m_id = aggregate_results(indist_7m)

    # ── In-distribution: train split (overfitting check) ──
    print("\n[3b] Evaluacion TRAIN split (overfitting check)...")
    print(f"  10 muestras de 2048 bytes")

    print("\n  --- 3.5M ---")
    train_35m = evaluate_corpus_split(
        model_35m, train_data, max_samples=10, sample_size=2048, label="3.5M-TR")
    agg_35m_tr = aggregate_results(train_35m)

    print("\n  --- 7M ---")
    train_7m = evaluate_corpus_split(
        model_7m, train_data, max_samples=10, sample_size=2048, label="7M-TR")
    agg_7m_tr = aggregate_results(train_7m)

    # ── Out-of-distribution ──
    print(f"\n[4/5] Evaluacion OUT-OF-DISTRIBUTION ({len(OOD_SAMPLES)} muestras)...")

    print("\n  --- 3.5M ---")
    ood_35m = evaluate_ood_samples(model_35m, OOD_SAMPLES, label="3.5M-OOD")
    agg_35m_ood = aggregate_results(list(ood_35m.values()))

    print("\n  --- 7M ---")
    ood_7m = evaluate_ood_samples(model_7m, OOD_SAMPLES, label="7M-OOD")
    agg_7m_ood = aggregate_results(list(ood_7m.values()))

    # ── Report ──
    print("\n" + "=" * 70)
    print("[5/5] RESULTADOS")
    print("=" * 70)

    print("\n## Resumen Agregado")
    print(f"{'':30s} {'3.5M':>12s} {'7M':>12s} {'Delta':>12s}")
    print("-" * 66)

    def row(label, v35, v7, fmt=".3f", lower_better=True):
        delta = v7 - v35
        sign = "+" if delta > 0 else ""
        better = "7M" if (delta < 0) == lower_better else "3.5M"
        print(f"{label:30s} {v35:>12{fmt}} {v7:>12{fmt}} {sign}{delta:>11{fmt}} ({better})")

    print("\nIn-Distribution (test split):")
    row("  BPB (weighted)", agg_35m_id["weighted_bpb"], agg_7m_id["weighted_bpb"])
    row("  BPB (mean)", agg_35m_id["mean_bpb"], agg_7m_id["mean_bpb"])
    row("  BPB (std)", agg_35m_id["std_bpb"], agg_7m_id["std_bpb"])
    row("  Accuracy top-1", agg_35m_id["mean_accuracy"], agg_7m_id["mean_accuracy"], lower_better=False)

    print("\nTrain split (overfitting check):")
    row("  BPB (weighted)", agg_35m_tr["weighted_bpb"], agg_7m_tr["weighted_bpb"])
    row("  Accuracy top-1", agg_35m_tr["mean_accuracy"], agg_7m_tr["mean_accuracy"], lower_better=False)

    print("\nOut-of-Distribution:")
    row("  BPB (weighted)", agg_35m_ood["weighted_bpb"], agg_7m_ood["weighted_bpb"])
    row("  BPB (mean)", agg_35m_ood["mean_bpb"], agg_7m_ood["mean_bpb"])
    row("  BPB (std)", agg_35m_ood["std_bpb"], agg_7m_ood["std_bpb"])
    row("  Accuracy top-1", agg_35m_ood["mean_accuracy"], agg_7m_ood["mean_accuracy"], lower_better=False)

    print("\nGeneralization gap (OOD - ID):")
    gap_35m = agg_35m_ood["weighted_bpb"] - agg_35m_id["weighted_bpb"]
    gap_7m = agg_7m_ood["weighted_bpb"] - agg_7m_id["weighted_bpb"]
    print(f"  3.5M: {gap_35m:+.3f} BPB")
    print(f"  7M:   {gap_7m:+.3f} BPB")
    print(f"  Mejor generalizacion: {'7M' if abs(gap_7m) < abs(gap_35m) else '3.5M'}")

    print("\nOverfitting gap (train - test):")
    ovfit_35m = agg_35m_tr["weighted_bpb"] - agg_35m_id["weighted_bpb"]
    ovfit_7m = agg_7m_tr["weighted_bpb"] - agg_7m_id["weighted_bpb"]
    print(f"  3.5M: {ovfit_35m:+.3f} BPB")
    print(f"  7M:   {ovfit_7m:+.3f} BPB")

    # ── OOD per-sample breakdown ──
    print("\n## OOD Per-Sample Breakdown")
    print(f"{'Sample':30s} {'3.5M BPB':>10s} {'7M BPB':>10s} {'Delta':>10s} {'Mejor':>8s}")
    print("-" * 70)
    for name in OOD_SAMPLES:
        if name in ood_35m and name in ood_7m:
            b35 = ood_35m[name]["bpb"]
            b7 = ood_7m[name]["bpb"]
            delta = b7 - b35
            mejor = "7M" if delta < 0 else "3.5M"
            print(f"{name:30s} {b35:>10.3f} {b7:>10.3f} {delta:>+10.3f} {mejor:>8s}")

    print(f"\n{'Accuracy':30s} {'3.5M':>10s} {'7M':>10s}")
    print("-" * 52)
    for name in OOD_SAMPLES:
        if name in ood_35m and name in ood_7m:
            a35 = ood_35m[name]["accuracy_top1"]
            a7 = ood_7m[name]["accuracy_top1"]
            print(f"{name:30s} {a35:>10.1%} {a7:>10.1%}")

    # ── Theoretical comparison ──
    print("\n## Comparacion con cotas teoricas")
    print(f"  Shannon floor (Python, infinito):  ~0.3-0.5 BPB")
    print(f"  Floor practico (3.5M params):      ~0.6-0.7 BPB")
    print(f"  3.5M ID actual:                    {agg_35m_id['weighted_bpb']:.3f} BPB")
    print(f"  7M ID actual:                      {agg_7m_id['weighted_bpb']:.3f} BPB")
    print(f"  Gap 3.5M vs floor practico:        {agg_35m_id['weighted_bpb'] - 0.65:+.3f} BPB")
    print(f"  Gap 7M vs floor practico:          {agg_7m_id['weighted_bpb'] - 0.65:+.3f} BPB")
    print(f"  Trivial (sin compresion):          8.000 BPB")
    print(f"  Compresion efectiva 3.5M:          {8.0 / agg_35m_id['weighted_bpb']:.1f}:1")
    print(f"  Compresion efectiva 7M:            {8.0 / agg_7m_id['weighted_bpb']:.1f}:1")

    # Save raw results as JSON
    output = {
        "models": {
            "3.5M": {"ckpt": CKPT_35M, "d": 512, "params": 3520816},
            "7M": {"ckpt": CKPT_7M, "d": 1024, "params": 7041328},
        },
        "in_distribution": {
            "3.5M": agg_35m_id, "7M": agg_7m_id,
        },
        "train_split": {
            "3.5M": agg_35m_tr, "7M": agg_7m_tr,
        },
        "out_of_distribution": {
            "3.5M": agg_35m_ood, "7M": agg_7m_ood,
        },
        "ood_per_sample": {
            "3.5M": {k: v for k, v in ood_35m.items()},
            "7M": {k: v for k, v in ood_7m.items()},
        },
        "generalization_gap": {"3.5M": gap_35m, "7M": gap_7m},
    }
    json_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResultados guardados en: {json_path}")

    print("\n" + "=" * 70)
    print("EVALUACION COMPLETA")
    print("=" * 70)


if __name__ == "__main__":
    main()
