"""Auto-upload new checkpoints to HuggingFace as they appear."""
import os
import time
import glob

CKPT_DIR = "/workspace/checkpoints_7m_200mb"
REPO_ID = "DavidCaastro/flux-lm-7m"
CHECK_INTERVAL = 300  # check every 5 minutes
UPLOADED = set()

def tlog(msg):
    t = time.time()
    ts = time.strftime("%H:%M:%S", time.localtime(t))
    print(f"[{ts}] [HF-UPLOAD] {msg}", flush=True)

def get_checkpoints():
    return sorted(glob.glob(os.path.join(CKPT_DIR, "flux_epoch_*.pt")))

def upload(path):
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi
    api = HfApi()
    fname = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1024**2
    tlog(f"Subiendo {fname} ({size_mb:.1f} MB)...")
    t0 = time.time()
    try:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"checkpoints/{fname}",
            repo_id=REPO_ID,
            commit_message=f"{fname} — 7M model, 200MB corpus"
        )
        tlog(f"OK: {fname} subido en {time.time()-t0:.0f}s")
        return True
    except Exception as e:
        tlog(f"ERROR subiendo {fname}: {e}")
        return False

if __name__ == "__main__":
    tlog(f"Iniciando auto-upload. Dir={CKPT_DIR}, Repo={REPO_ID}")
    tlog(f"Intervalo de chequeo: {CHECK_INTERVAL}s")

    # Mark existing checkpoints as already handled (epoch 5 already uploaded)
    for p in get_checkpoints():
        UPLOADED.add(p)
        tlog(f"Ya existente (skip): {os.path.basename(p)}")

    while True:
        time.sleep(CHECK_INTERVAL)
        for p in get_checkpoints():
            if p not in UPLOADED:
                # Wait 2 min for file to consolidate
                tlog(f"Nuevo checkpoint detectado: {os.path.basename(p)}, esperando 120s...")
                time.sleep(120)
                if upload(p):
                    UPLOADED.add(p)
                else:
                    # Retry once after 60s
                    tlog("Reintentando en 60s...")
                    time.sleep(60)
                    if upload(p):
                        UPLOADED.add(p)
