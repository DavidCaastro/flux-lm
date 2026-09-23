"""
Graceful shutdown: wait for epoch 21, stop training, upload to HF, cleanup instance.
Run on Vast.ai instance: python -u graceful_shutdown.py > /workspace/shutdown.log 2>&1 &
"""
import os
import sys
import time
import glob
import signal
import subprocess

TARGET_EPOCH = 20  # last epoch with auto-checkpoint (multiple of 5)
WAIT_EXTRA = True   # after target, let training run 1 more epoch if possible
CKPT_DIR = "/workspace/checkpoints_7m_200mb"
TRAIN_LOG = "/workspace/train_7m_200mb.log"
REPO_ID = "DavidCaastro/flux-lm-7m"
CHECK_INTERVAL = 60  # check every 60s

def tlog(msg):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] [SHUTDOWN] {msg}", flush=True)

def get_latest_epoch():
    """Parse training log to find last completed epoch."""
    try:
        result = subprocess.run(
            ["grep", "EPOCH.*COMPLETADA", TRAIN_LOG],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if lines and lines[-1]:
            # Extract epoch number from "EPOCH X COMPLETADA"
            for part in lines[-1].split():
                try:
                    return int(part)
                except ValueError:
                    continue
    except Exception as e:
        tlog(f"Error leyendo log: {e}")
    return 0

def kill_training():
    """Kill training process and auto_upload_hf daemon."""
    tlog("Matando procesos de entrenamiento...")
    # Kill train.py
    os.system("pkill -f 'python.*train.py' 2>/dev/null")
    time.sleep(2)
    # Kill auto_upload_hf.py
    os.system("pkill -f 'auto_upload_hf' 2>/dev/null")
    time.sleep(2)
    # Verify
    result = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True)
    if result.returncode == 0:
        tlog("WARN: train.py aun vivo, enviando SIGKILL...")
        os.system("pkill -9 -f 'python.*train.py' 2>/dev/null")
        time.sleep(2)
    tlog("Procesos matados OK")

def upload_checkpoint(epoch):
    """Upload specific checkpoint to HuggingFace."""
    ckpt_path = os.path.join(CKPT_DIR, f"flux_epoch_{epoch:04d}.pt")
    if not os.path.exists(ckpt_path):
        tlog(f"ERROR: checkpoint no encontrado: {ckpt_path}")
        return False

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi
    api = HfApi()

    fname = os.path.basename(ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / 1024**2
    tlog(f"Subiendo {fname} ({size_mb:.1f} MB) a {REPO_ID}...")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"checkpoints/{fname}",
                repo_id=REPO_ID,
                commit_message=f"{fname} — 7M model, 200MB corpus (final checkpoint)"
            )
            tlog(f"OK: {fname} subido en {time.time()-t0:.0f}s (intento {attempt})")
            return True
        except Exception as e:
            tlog(f"ERROR intento {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                tlog(f"Reintentando en 120s...")
                time.sleep(120)

    tlog("FALLO: no se pudo subir tras 3 intentos")
    return False

def upload_final_log():
    """Upload training log to HF for reference."""
    if not os.path.exists(TRAIN_LOG):
        return
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi
    api = HfApi()
    tlog("Subiendo log de entrenamiento...")
    try:
        api.upload_file(
            path_or_fileobj=TRAIN_LOG,
            path_in_repo="logs/train_7m_200mb.log",
            repo_id=REPO_ID,
            commit_message="training log — 7M model, 200MB corpus"
        )
        tlog("Log subido OK")
    except Exception as e:
        tlog(f"WARN: no se pudo subir log: {e}")

def cleanup_instance():
    """Remove all training data from the instance."""
    tlog("=== LIMPIEZA TOTAL DE LA INSTANCIA ===")

    dirs_to_remove = [
        "/workspace/checkpoints_7m_200mb",
        "/workspace/flux-lm",
        "/workspace/checkpoints",
    ]
    files_to_remove = [
        "/workspace/train_7m_200mb.log",
        "/workspace/auto_upload_hf.py",
        "/workspace/auto_upload_hf.log",
        "/workspace/shutdown.log",
        "/workspace/corpus_python_200mb.txt",
        "/workspace/corpus_python.txt",
    ]

    for d in dirs_to_remove:
        if os.path.exists(d):
            tlog(f"Eliminando directorio: {d}")
            os.system(f"rm -rf {d}")

    for f in files_to_remove:
        if os.path.exists(f):
            tlog(f"Eliminando archivo: {f}")
            os.remove(f)

    # Clean anything else in /workspace
    tlog("Limpiando todo lo restante en /workspace...")
    os.system("rm -rf /workspace/* /workspace/.* 2>/dev/null")
    tlog("Limpieza completada")

if __name__ == "__main__":
    tlog(f"Iniciando graceful shutdown daemon")
    tlog(f"Target: epoch {TARGET_EPOCH}")
    tlog(f"Checkpoint dir: {CKPT_DIR}")
    tlog(f"Repo HF: {REPO_ID}")

    current = get_latest_epoch()
    tlog(f"Epoch actual: {current}")

    if current >= TARGET_EPOCH:
        tlog(f"Epoch {TARGET_EPOCH} ya completada! Procediendo...")
    else:
        tlog(f"Esperando epoch {TARGET_EPOCH} (faltan {TARGET_EPOCH - current} epochs)...")
        while True:
            time.sleep(CHECK_INTERVAL)
            current = get_latest_epoch()
            if current >= TARGET_EPOCH:
                tlog(f"Epoch {TARGET_EPOCH} COMPLETADA (actual: {current})")
                break
            tlog(f"Esperando... epoch actual: {current}/{TARGET_EPOCH}")

    # Step 1: Wait for checkpoint file to appear on disk
    tlog("Esperando checkpoint en disco...")
    ckpt_target = os.path.join(CKPT_DIR, f"flux_epoch_{TARGET_EPOCH:04d}.pt")
    for _ in range(60):  # max 60s wait
        if os.path.exists(ckpt_target):
            break
        time.sleep(1)
    time.sleep(5)  # extra settle time

    # Step 1b: Let training run one more epoch if WAIT_EXTRA
    if WAIT_EXTRA:
        extra_target = TARGET_EPOCH + 1
        tlog(f"Dejando correr 1 epoch extra hasta epoch {extra_target}...")
        while True:
            time.sleep(CHECK_INTERVAL)
            current = get_latest_epoch()
            if current >= extra_target:
                tlog(f"Epoch extra {extra_target} COMPLETADA")
                break
            tlog(f"Esperando epoch extra... actual: {current}/{extra_target}")

    # Step 2: Kill training
    kill_training()

    # Step 3: Upload all checkpoints not yet in HF
    all_ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "flux_epoch_*.pt")))
    # epochs 5, 10, 15 already uploaded by auto_upload_hf daemon
    already_uploaded = {5, 10, 15}
    for ckpt_path in all_ckpts:
        fname = os.path.basename(ckpt_path)
        epoch_num = int(fname.split("_")[-1].split(".")[0])
        if epoch_num not in already_uploaded:
            upload_checkpoint(epoch_num)

    # Step 5: Upload training log
    upload_final_log()

    # Step 6: Cleanup
    tlog("Todos los uploads completados. Iniciando limpieza en 60s...")
    time.sleep(60)
    cleanup_instance()

    tlog("=== SHUTDOWN COMPLETO ===")
    tlog("La instancia puede ser destruida manualmente desde Vast.ai")
