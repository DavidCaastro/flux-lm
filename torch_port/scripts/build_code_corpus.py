#!/usr/bin/env python3
"""Build a Python code corpus from popular GitHub repos.

Downloads top Python repos as zip archives, extracts .py files,
filters out junk (tests, configs, generated), and concatenates
into a single UTF-8 text file.

Usage:
    python3 scripts/build_code_corpus.py --output data/corpus_python.txt --target-mb 60
    python3 scripts/build_code_corpus.py --output data/corpus_python_1gb.txt --target-mb 1024
"""

import argparse
import io
import os
import zipfile
import urllib.request
import time

# Large, high-quality Python repos organized by category.
# Ordered roughly by code volume to reach target faster.
REPOS = [
    # === Large codebases (10-50+ MB of .py each) ===
    ("python/cpython", "main"),
    ("pytorch/pytorch", "main"),
    ("tensorflow/tensorflow", "master"),
    ("scikit-learn/scikit-learn", "main"),
    ("pandas-dev/pandas", "main"),
    ("numpy/numpy", "main"),
    ("matplotlib/matplotlib", "main"),
    ("scipy/scipy", "main"),
    ("sympy/sympy", "master"),
    ("huggingface/transformers", "main"),
    ("ansible/ansible", "devel"),
    ("saltstack/salt", "master"),
    ("odoo/odoo", "master"),
    ("home-assistant/core", "dev"),

    # === Web frameworks ===
    ("django/django", "main"),
    ("pallets/flask", "main"),
    ("fastapi/fastapi", "master"),
    ("tornadoweb/tornado", "master"),
    ("encode/starlette", "master"),
    ("huge-success/sanic", "main"),
    ("bottlepy/bottle", "master"),
    ("falconry/falcon", "master"),
    ("channelcat/sanic", "main"),

    # === ML / AI ===
    ("keras-team/keras", "master"),
    ("openai/gym", "master"),
    ("ray-project/ray", "master"),
    ("apache/airflow", "main"),
    ("mlflow/mlflow", "master"),
    ("dmlc/xgboost", "master"),
    ("Lightning-AI/pytorch-lightning", "master"),
    ("huggingface/diffusers", "main"),
    ("huggingface/datasets", "main"),
    ("langchain-ai/langchain", "master"),
    ("vllm-project/vllm", "main"),
    ("openai/whisper", "main"),

    # === Data / DB ===
    ("apache/spark", "master"),
    ("great-expectations/great_expectations", "develop"),
    ("dagster-io/dagster", "master"),
    ("prefecthq/prefect", "main"),
    ("dbt-labs/dbt-core", "main"),
    ("sqlalchemy/sqlalchemy", "main"),
    ("coleifer/peewee", "master"),
    ("tortoise/tortoise-orm", "develop"),

    # === DevOps / CLI ===
    ("docker/compose", "main"),
    ("python-poetry/poetry", "main"),
    ("pypa/pip", "main"),
    ("pallets/click", "main"),
    ("tqdm/tqdm", "master"),
    ("psf/black", "main"),
    ("PyCQA/pylint", "main"),
    ("PyCQA/flake8", "main"),
    ("astral-sh/ruff", "main"),
    ("pre-commit/pre-commit", "main"),
    ("pypa/setuptools", "main"),
    ("pypa/virtualenv", "main"),

    # === Networking / HTTP ===
    ("psf/requests", "main"),
    ("aio-libs/aiohttp", "master"),
    ("encode/httpx", "master"),
    ("urllib3/urllib3", "main"),
    ("scrapy/scrapy", "master"),
    ("MagicStack/uvloop", "master"),
    ("encode/uvicorn", "master"),
    ("celery/celery", "main"),
    ("benoitc/gunicorn", "master"),
    ("gevent/gevent", "master"),
    ("twisted/twisted", "trunk"),

    # === Security / Crypto ===
    ("pyca/cryptography", "main"),
    ("paramiko/paramiko", "main"),
    ("mitmproxy/mitmproxy", "main"),
    ("certbot/certbot", "master"),

    # === Data science / Viz ===
    ("bokeh/bokeh", "branch-3.7"),
    ("plotly/plotly.py", "master"),
    ("streamlit/streamlit", "develop"),
    ("gradio-app/gradio", "main"),
    ("Textualize/rich", "master"),
    ("Textualize/textual", "main"),

    # === Utilities ===
    ("pydantic/pydantic", "main"),
    ("python-attrs/attrs", "main"),
    ("more-itertools/more-itertools", "master"),
    ("pytoolz/toolz", "master"),
    ("jazzband/tablib", "master"),
    ("arrow-py/arrow", "master"),
    ("dateutil/dateutil", "master"),
    ("simplejson/simplejson", "master"),
    ("yaml/pyyaml", "main"),
    ("msgpack/msgpack-python", "main"),

    # === Testing ===
    ("pytest-dev/pytest", "main"),
    ("HypothesisWorks/hypothesis", "master"),
    ("robotframework/robotframework", "master"),
    ("locustio/locust", "master"),

    # === AWS / Cloud ===
    ("boto/boto3", "develop"),
    ("aws/aws-cli", "v2"),
    ("localstack/localstack", "master"),
    ("pulumi/pulumi", "master"),

    # === NLP / Text ===
    ("nltk/nltk", "develop"),
    ("explosion/spaCy", "master"),
    ("RasaHQ/rasa", "main"),
    ("flairNLP/flair", "master"),
    ("stanfordnlp/stanza", "main"),

    # === Image / CV ===
    ("python-pillow/Pillow", "main"),
    ("opencv/opencv-python", "4.x"),
    ("ultralytics/ultralytics", "main"),
    ("facebookresearch/detectron2", "main"),

    # === Misc large projects ===
    ("pallets/werkzeug", "main"),
    ("pallets/jinja", "main"),
    ("pallets/markupsafe", "main"),
    ("mwaskom/seaborn", "master"),
    ("networkx/networkx", "main"),
    ("giampaolo/psutil", "master"),
    ("docker/docker-py", "main"),
    ("fabric/fabric", "main"),
    ("pyinstaller/pyinstaller", "develop"),
    ("sphinx-doc/sphinx", "master"),
    ("readthedocs/readthedocs.org", "main"),
    ("jupyter/notebook", "main"),
    ("ipython/ipython", "main"),
    ("jupyterlab/jupyterlab", "main"),
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
MAX_FILE_BYTES = 200_000


def should_skip_path(path: str) -> bool:
    parts = path.lower().split("/")
    for part in parts[:-1]:
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
        with urllib.request.urlopen(req, timeout=180) as resp:
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

            print(f"    → {len(files)} archivos, total acumulado: "
                  f"{total_bytes/1024/1024:.1f} MB")
            del zip_data, files
            time.sleep(1)

    final_size = os.path.getsize(args.output)
    print(f"\nCorpus final: {total_files} archivos, {final_size/1024/1024:.1f} MB")
    print("Listo!")


if __name__ == "__main__":
    main()
