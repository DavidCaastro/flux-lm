#!/usr/bin/env python3
"""Build a multilingual UTF-8 corpus from Wikipedia dumps for Flux byte-level LM.

Downloads compressed Wikipedia article dumps (static files, no API rate limits),
streams decompression, extracts plaintext, and builds a single UTF-8 corpus.

Usage:
    python3 scripts/build_wiki_corpus.py                          # default ~100MB
    python3 scripts/build_wiki_corpus.py --target-mb 1000         # 1GB corpus
    python3 scripts/build_wiki_corpus.py --target-mb 50 --out data/corpus_50mb.txt
"""

import argparse
import bz2
import math
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from io import BytesIO

# Languages with diverse scripts — (code, weight)
LANGUAGES = [
    # Latin-based
    ("en", 20),  ("es", 12),  ("fr", 10),  ("de", 10),
    ("pt", 8),   ("pl", 6),   ("tr", 5),   ("vi", 8),
    ("cs", 4),   ("ro", 4),
    # Cyrillic
    ("ru", 10),  ("uk", 5),   ("sr", 3),   ("bg", 3),
    # CJK
    ("zh", 12),  ("ja", 12),  ("ko", 8),
    # Arabic script
    ("ar", 8),   ("fa", 5),   ("ur", 3),
    # Indic
    ("hi", 6),   ("bn", 4),   ("ta", 3),   ("te", 3),   ("th", 5),
    # Other scripts
    ("he", 5),   ("el", 5),   ("ka", 3),   ("hy", 3),
    ("my", 2),   ("km", 2),   ("si", 2),   ("am", 2),
]
TOTAL_WEIGHT = sum(w for _, w in LANGUAGES)

UA = "FluxLM-CorpusBuilder/1.0 (byte-level LM research; github.com/flux-lm)"


def strip_wiki_markup(text: str) -> str:
    """Remove MediaWiki markup and return clean plaintext."""
    # Remove templates {{...}} (greedy nested removal)
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif text[i:i+2] == '}}' and depth > 0:
            depth -= 1
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    text = ''.join(result)

    # Remove HTML tags
    text = re.sub(r'<[^>]+/?>', '', text)
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Convert wiki links [[target|display]] -> display, [[target]] -> target
    text = re.sub(r'\[\[[^\]]*\|([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    # Remove external links [url text] -> text
    text = re.sub(r'\[https?://[^\s\]]*\s*([^\]]*)\]', r'\1', text)
    # Remove refs
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    # Remove bold/italic markers
    text = re.sub(r"'{2,5}", '', text)
    # Remove section headers markup but keep text
    text = re.sub(r'^=+\s*(.*?)\s*=+\s*$', r'\1', text, flags=re.MULTILINE)
    # Remove category/file/image links
    text = re.sub(
        r'\[\[(Category|File|Image|Archivo|Categoría|Fichier|Catégorie|'
        r'Datei|Kategorie|Файл|Категория|分类|カテゴリ):[^\]]+\]\]',
        '', text, flags=re.IGNORECASE)
    # Remove tables {| ... |}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove remaining markup chars
    text = re.sub(r'[|\[\]{}]', '', text)
    # Collapse whitespace
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def download_dump_chunk(lang: str, target_bytes: int) -> bytes:
    """Download and extract text from a Wikipedia dump until target_bytes reached."""
    dump_url = (
        f"https://dumps.wikimedia.org/{lang}wiki/latest/"
        f"{lang}wiki-latest-pages-articles.xml.bz2"
    )

    print(f"  [{lang}] downloading from dumps.wikimedia.org ...", flush=True)

    req = urllib.request.Request(dump_url, headers={"User-Agent": UA})

    collected = bytearray()
    articles = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            decompressor = bz2.BZ2Decompressor()
            xml_buffer = b""
            in_text = False
            current_text = []
            chunk_size = 64 * 1024  # 64KB read chunks

            while len(collected) < target_bytes:
                compressed = resp.read(chunk_size)
                if not compressed:
                    break

                try:
                    decompressed = decompressor.decompress(compressed)
                except EOFError:
                    # BZ2 stream ended, start new decompressor for next block
                    decompressor = bz2.BZ2Decompressor()
                    try:
                        decompressed = decompressor.decompress(compressed)
                    except Exception:
                        break

                xml_buffer += decompressed

                # Simple streaming XML parse: find <text ...>content</text> blocks
                while True:
                    if not in_text:
                        match = re.search(rb'<text[^>]*>', xml_buffer)
                        if match is None:
                            xml_buffer = xml_buffer[-200:] if len(xml_buffer) > 200 else xml_buffer
                            break
                        xml_buffer = xml_buffer[match.end():]
                        in_text = True
                        current_text = []

                    if in_text:
                        end_idx = xml_buffer.find(b'</text>')
                        if end_idx == -1:
                            if len(xml_buffer) > 500_000:
                                in_text = False
                                xml_buffer = xml_buffer[-200:]
                            else:
                                current_text.append(xml_buffer)
                                xml_buffer = b""
                            break

                        current_text.append(xml_buffer[:end_idx])
                        xml_buffer = xml_buffer[end_idx + 7:]
                        in_text = False

                        raw = b"".join(current_text)
                        try:
                            wiki_text = raw.decode("utf-8", errors="replace")
                        except Exception:
                            continue

                        if (wiki_text.startswith("#REDIRECT") or
                            wiki_text.startswith("#redirect") or
                            len(wiki_text) < 500):
                            continue

                        clean = strip_wiki_markup(wiki_text)
                        if len(clean) < 200:
                            continue

                        encoded = clean.encode("utf-8")
                        collected.extend(encoded)
                        collected.extend(b"\n\n")
                        articles += 1

                        if articles % 50 == 0:
                            mb = len(collected) / 1024 / 1024
                            target_mb = target_bytes / 1024 / 1024
                            print(f"  [{lang}] {articles} articles, "
                                  f"{mb:.1f}/{target_mb:.1f} MB", flush=True)

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  [{lang}] download error: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"  [{lang}] error: {e}", file=sys.stderr, flush=True)

    print(f"  [{lang}] done: {articles} articles, "
          f"{len(collected)/1024/1024:.1f} MB", flush=True)
    return bytes(collected[:target_bytes])


def build_corpus(target_mb: float, output_path: str):
    target_bytes = int(target_mb * 1024 * 1024)

    print(f"Target: {target_mb:.0f} MB ({target_bytes:,} bytes)")
    print(f"Languages: {len(LANGUAGES)}")
    print(f"Output: {output_path}")
    print(f"Source: Wikipedia database dumps (streaming bz2)")
    print()

    lang_targets = {
        lang: int(target_bytes * weight / TOTAL_WEIGHT)
        for lang, weight in LANGUAGES
    }

    results = {}
    total_fetched = 0

    for lang, byte_target in lang_targets.items():
        chunk = download_dump_chunk(lang, byte_target)
        results[lang] = chunk
        total_fetched += len(chunk)
        pct = total_fetched / target_bytes * 100
        print(f"  >>> progress: {total_fetched/1024/1024:.1f} / "
              f"{target_mb:.0f} MB ({pct:.0f}%)\n", flush=True)

    # Shuffle at paragraph level for mixing
    print("Shuffling and writing ...", flush=True)
    all_paragraphs = []
    for lang_code, chunk in results.items():
        text = chunk.decode("utf-8", errors="replace")
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
        all_paragraphs.extend(paragraphs)

    random.seed(42)
    random.shuffle(all_paragraphs)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        for para in all_paragraphs:
            f.write(para.encode("utf-8"))
            f.write(b"\n\n")

    final_size = os.path.getsize(output_path)

    # Stats
    with open(output_path, "rb") as f:
        data = f.read()
    unique_bytes = len(set(data))
    bc = Counter(data)
    entropy = -sum((c / len(data)) * math.log2(c / len(data)) for c in bc.values())
    ascii_pct = sum(c for b, c in bc.items() if b < 128) / len(data) * 100

    print(f"\n{'='*55}")
    print(f" Corpus: {output_path}")
    print(f" Size:          {final_size/1024/1024:.1f} MB ({final_size:,} bytes)")
    print(f" Paragraphs:    {len(all_paragraphs):,}")
    print(f" Unique bytes:  {unique_bytes}/256")
    print(f" Entropy:       {entropy:.3f} bits/byte")
    print(f" ASCII:         {ascii_pct:.1f}%")
    print(f" Multi-byte:    {100-ascii_pct:.1f}%")
    print(f"{'='*55}")


def main():
    p = argparse.ArgumentParser(
        description="Build multilingual UTF-8 corpus from Wikipedia dumps")
    p.add_argument("--target-mb", type=float, default=100,
                   help="Target corpus size in MB (default: 100)")
    p.add_argument("--out", type=str, default=None,
                   help="Output file path")
    args = p.parse_args()

    if args.out is None:
        args.out = f"data/corpus_wiki_{int(args.target_mb)}mb.txt"

    build_corpus(args.target_mb, args.out)


if __name__ == "__main__":
    main()
