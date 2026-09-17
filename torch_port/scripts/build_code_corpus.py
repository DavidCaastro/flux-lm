#!/usr/bin/env python3
"""Build a ~50-80MB Python code corpus from popular GitHub repos.

Downloads top Python repos as zip archives, extracts .py files,
filters out junk (tests, configs, generated), and concatenates
into a single UTF-8 text file.

Usage:
    python3 scripts/build_code_corpus.py --output data/corpus_python.txt --target-mb 60
"""

import argparse
import io
import os
import zipfile
import urllib.request
import time

# Repos: popular, diverse, high-quality Python code
REPOS = [
    # Web frameworks
    ("django/django", "main"),
    ("pallets/flask", "main"),
    ("fastapi/fastapi", "master"),
    ("encode/starlette", "master"),
    ("encode/httpx", "master"),
    ("huge-success/sanic", "main"),
    # ML / Data (medium-sized repos, NOT pytorch/cpython)
    ("scikit-learn/scikit-learn", "main"),
    ("pandas-dev/pandas", "main"),
    ("numpy/numpy", "main"),
    ("huggingface/transformers", "main"),
    # Utilities
    ("psf/requests", "main"),
    ("aio-libs/aiohttp", "master"),
    ("pydantic/pydantic", "main"),
    ("python-attrs/attrs", "main"),
    ("more-itertools/more-itertools", "master"),
    # CLI / DevOps
    ("pallets/click", "main"),
    ("python-poetry/poetry", "main"),
    ("pypa/pip", "main"),
    ("pre-commit/pre-commit", "main"),
    # Async / networking
    ("MagicStack/uvloop", "master"),
    ("encode/uvicorn", "master"),
    ("celery/celery", "main"),
    # Data / parsing
    ("yaml/pyyaml", "main"),
    ("simplejson/simplejson", "master"),
    ("jmespath/jmespath.py", "develop"),
    ("boto/boto3", "develop"),
]

SKIP_DIRS = {
    "test", "tests", "testing", "test_", "_test",
    "vendor", "vendored", "third_party", "thirdparty",
    "__pycache__", ".git", "node_modules", "dist", "build",
    "docs", "doc", "examples", "benchmarks", "fixtures",
    "migrations", "locale", "translations",
}

SKIP_SUFFIXES = (
    "_test.py", "test_.py", "_tests.py",
    "conftest.py", "setup.py", "setup.cfg",
    "__main__.py",
)

MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 100_000  # skip huge generated files


def should_skip_path(path: str) -> bool:
    parts = path.lower().split("/")
    for part in parts[:-1]:  # directories
        if any(skip in part for skip in SKIP_DIRS):
            return True
    fname = parts[-1]
    if not fname.endswith(".py"):
        return True
    if fname.startswith("test_") or fname.endswith(tuple(SKIP_SUFFIXES)):
        return True
    return False


def download_repo(owner_repo: str, branch: str) -> bytes | None:
    url = f"https://github.com/{owner_repo}/archive/refs/heads/{branch}.zip"
    print(f"  Descargando {owner_repo} ({branch})...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "flux-lm-corpus-builder"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        print(f"{len(data)/1024/1024:.1f} MB")
        return data
    except Exception as e:
        print(f"FALLÓ: {e}")
        return None


def extract_python_files(zip_data: bytes, target_mb: float, current_size: int) -> list[str]:
    target_bytes = int(target_mb * 1024 * 1024)
    files = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for info in zf.infolist():
                if current_size >= target_bytes:
                    break
                if info.is_dir():
                    continue
                if should_skip_path(info.filename):
                    continue
                if info.file_size < MIN_FILE_BYTES or info.file_size > MAX_FILE_BYTES:
                    continue
                try:
                    content = zf.read(info.filename).decode("utf-8", errors="ignore")
                    # Basic quality filter: must have at least some code
                    if "def " not in content and "class " not in content:
                        continue
                    files.append(content)
                    current_size += len(content.encode("utf-8"))
                except Exception:
                    continue
    except zipfile.BadZipFile:
        print("  (zip corrupto, saltando)")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/corpus_python.txt")
    parser.add_argument("--target-mb", type=float, default=60.0)
    args = parser.parse_args()

    print(f"Objetivo: {args.target_mb} MB de código Python")
    print(f"Salida: {args.output}")
    print(f"Repos: {len(REPOS)}")
    print()

    total_bytes = 0
    total_files = 0
    target_bytes = int(args.target_mb * 1024 * 1024)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Write incrementally to avoid memory issues and survive disconnects
    with open(args.output, "w", encoding="utf-8") as out:
        for owner_repo, branch in REPOS:
            if total_bytes >= target_bytes:
                print(f"  Objetivo alcanzado ({total_bytes/1024/1024:.1f} MB)")
                break

            zip_data = download_repo(owner_repo, branch)
            if zip_data is None:
                time.sleep(2)
                continue

            files = extract_python_files(zip_data, args.target_mb, total_bytes)
            for content in files:
                out.write(content)
                out.write("\n\n")
                total_bytes += len(content.encode("utf-8"))
                total_files += 1
                if total_bytes >= target_bytes:
                    break
            out.flush()

            print(f"    → {len(files)} archivos, total acumulado: {total_bytes/1024/1024:.1f} MB")
            del zip_data, files  # free memory
            time.sleep(1)

    final_size = os.path.getsize(args.output)
    print(f"\nCorpus final: {total_files} archivos, {final_size/1024/1024:.1f} MB")
    print("Listo!")


if __name__ == "__main__":
    main()
