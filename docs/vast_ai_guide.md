# Guia Operativa — Vast.ai para Entrenamiento GPU

> Documentacion basada en experiencia real con RTX 4090 en Vast.ai para el proyecto Flux LM.
> Ultima actualizacion: 2026-09-21

---

## 1. Conceptos Basicos

Vast.ai es un marketplace de GPUs en la nube. Proveedores independientes ofrecen sus GPUs y los usuarios las alquilan por hora. Las instancias corren Docker containers con acceso SSH.

### Tipos de instancia

| Tipo | Precio | Interrupcion | Uso recomendado |
|------|--------|--------------|-----------------|
| **On-demand** | Precio base | Nunca | Runs cortos (<24h), debugging |
| **Reserved** | ~50% descuento | Nunca | Training largo (>24h) — RECOMENDADO |
| **Interruptible** | ~50%+ descuento | Posible si outbid | Solo con checkpoints frecuentes |

### Recursos compartidos

- **GPU**: Acceso exclusivo (nunca compartida)
- **CPU**: Proporcional a GPUs alquiladas (1 de 4 GPUs = 25% CPU), puede burst
- **RAM**: Proporcional. OOM killer si se excede
- **Disco**: Fijado al crear, NO se puede cambiar despues

---

## 2. Seleccion de GPU

### Requisitos minimos para deep learning (modelos ~3-10M params)

| Recurso | Minimo | Ideal |
|---------|--------|-------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB, sm_89) |
| RAM | 16 GB | 32+ GB |
| CPU | 4 cores | 8+ cores |
| Disco | 40 GB | 100+ GB |
| Red | 100 Mbps | 500+ Mbps |
| Fiabilidad | >95% | >99% |

### Filtros CLI para buscar ofertas

```bash
# Instalar CLI
pip install vastai

# Configurar API key (obtener de https://cloud.vast.ai/account/)
vastai set api-key <TU_API_KEY>

# Buscar RTX 4090 con 24GB+, disco 40GB+, fiabilidad >95%
vastai search offers 'gpu_name=RTX 4090 num_gpus=1 gpu_ram>=24000 disk_space>=40 reliability>0.95' \
  --order 'dph_total'
```

### Filtros via API

```
GET /api/v0/bundles/
gpu_name: {"in": ["RTX 4090"]}
num_gpus: {"gte": 1}
gpu_ram: {"gte": 24000}
```

---

## 3. Creacion de Instancia

### Desde CLI

```bash
# Crear instancia on-demand
vastai create instance <OFFER_ID> \
  --image nvcr.io/nvidia/pytorch:26.08-py3 \
  --disk 50 \
  --onstart-cmd "touch ~/.no_auto_tmux"

# Crear instancia con script de setup
vastai create instance <OFFER_ID> \
  --image nvcr.io/nvidia/pytorch:26.08-py3 \
  --disk 50 \
  --onstart-cmd "bash /workspace/setup.sh"
```

### Parametros clave (API)

```
PUT /api/v0/asks/{offer_id}/
{
  "image": "nvcr.io/nvidia/pytorch:26.08-py3",
  "disk": 50,
  "runtype": "ssh_direct",
  "env": {"-p 8080:8080": "1"},
  "onstart": "touch ~/.no_auto_tmux",
  "python_utf8": true
}
```

### Docker Image recomendada

**NVIDIA NGC PyTorch nv26.08** — incluye todo lo necesario:
- Ubuntu 24.04, Python 3.12, PyTorch 2.14
- CUDA 13.4, cuDNN 9.25, NCCL 2.30
- nvcc, gcc 13.3, ninja 1.13 (JIT compilation OK)
- wandb preinstalado

> NOTA: Las imagenes NGC de NVIDIA son las mas completas. Evitar imagenes custom salvo necesidad especifica.

---

## 4. Conexion SSH

### Configuracion inicial (una sola vez)

```bash
# 1. Generar clave SSH (si no existe)
ssh-keygen -t ed25519 -C "email@example.com" -f ~/.ssh/id_vast

# 2. En Windows, la clave queda en: C:\Users\<User>\.ssh\id_vast

# 3. Subir clave publica a Vast.ai
#    https://cloud.vast.ai/manage-keys/
#    Copiar contenido de: cat ~/.ssh/id_vast.pub

# 4. Configurar alias SSH en ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'
Host vast
    HostName <IP_INSTANCIA>
    Port <PUERTO>
    User root
    IdentityFile ~/.ssh/id_vast
    StrictHostKeyChecking no
    ServerAliveInterval 30
    ServerAliveCountMax 10
EOF
```

### Conexion basica

```bash
# Conectar
ssh vast

# Con timeout (cuando hay inestabilidad)
ssh -o ConnectTimeout=15 vast

# Comando rapido sin sesion interactiva
ssh vast "nvidia-smi"

# Port forwarding (ej. TensorBoard)
ssh vast -L 6006:localhost:6006
```

### VS Code Remote SSH

1. Instalar extension "Remote - SSH"
2. Abrir paleta: `Remote-SSH: Connect to Host...`
3. Usar: `ssh -i ~/.ssh/id_vast -p <PUERTO> root@<IP>`
4. Workspace: `/workspace/`

### tmux (sesion por defecto)

Vast.ai lanza tmux automaticamente al conectar por SSH.

| Atajo | Accion |
|-------|--------|
| `Ctrl+B, C` | Nueva ventana |
| `Ctrl+B, N` | Siguiente ventana |
| `Ctrl+B, D` | Detach (desconectar sin matar) |
| `Ctrl+B, [` | Modo scroll (q para salir) |

```bash
# Desactivar auto-tmux (recomendado para SCP/rsync)
ssh vast "touch ~/.no_auto_tmux"

# Reactivar
ssh vast "rm ~/.no_auto_tmux"
```

---

## 5. Reconexion (cuando cambia IP/puerto)

Las instancias cambian de IP y puerto al reiniciarse o recrearse.

### Opcion 1: CLI (recomendada)

```bash
# Ver instancias activas con IP y puerto
vastai show instances

# La salida muestra: ID, status, ssh_host, ssh_port, gpu_name, etc.
# Actualizar ~/.ssh/config con los nuevos datos
```

### Opcion 2: Consola web

1. Ir a https://cloud.vast.ai/instances/
2. Copiar el comando SSH que aparece junto a la instancia
3. Actualizar `~/.ssh/config`

### Verificacion paso a paso

```bash
# 1. Ping basico (Windows)
ping -n 2 <NUEVA_IP>

# 2. SSH minimo
ssh vast "echo ok"

# 3. GPU status
ssh vast "nvidia-smi"

# 4. Procesos activos
ssh vast "ps aux | grep python"

# 5. Ultimo log
ssh vast "tail -30 /workspace/train*.log"
```

---

## 6. Transferencia de Archivos

### PROBLEMA CONOCIDO

Vast.ai **cierra conexiones SSH largas** durante transferencias de archivos grandes (>30 MB).
SCP y SFTP fallan con "Connection closed by remote host" a mitad de transferencia.

### Estrategias (ordenadas por fiabilidad)

#### A. GitHub Release como intermediario (MEJOR para archivos >30MB)

Usar GitHub Releases para transferir archivos grandes via la API, evitando las limitaciones de SSH.

```bash
# 1. Comprimir en vast
ssh vast "gzip -k -1 /workspace/archivo.pt"

# 2. Crear release desde local
gh release create nombre-release --title "Titulo" --notes "Descripcion"

# 3. Obtener release ID
RELEASE_ID=$(gh api repos/OWNER/REPO/releases/tags/nombre-release --jq '.id')

# 4. Subir desde vast (INTERACTIVO, no nohup)
GH_TOKEN=$(gh auth token)
ssh vast "curl -s -X POST \
  -H 'Authorization: token $GH_TOKEN' \
  -H 'Content-Type: application/gzip' \
  --data-binary @/workspace/archivo.pt.gz \
  'https://uploads.github.com/repos/OWNER/REPO/releases/$RELEASE_ID/assets?name=archivo.pt.gz'"

# 5. Descargar a local
gh release download nombre-release --pattern "archivo.pt.gz" --dir ./destino/

# 6. Descomprimir y verificar
gzip -dk ./destino/archivo.pt.gz
md5sum ./destino/archivo.pt
```

**Lecciones aprendidas:**
- Los uploads DEBEN ser interactivos (SSH directo). `nohup` en vast **no transmite datos** — curl queda atascado a 0 KB/s
- Velocidad tipica de upload (vast -> GitHub): **100-200 KB/s** (~6-10 min por 60 MB)
- `gzip -1` (compresion rapida) reduce ~25% el tamanio de checkpoints PyTorch
- Verificar siempre MD5 despues de descargar (`md5sum` en ambos lados)
- Limpiar assets en estado `starter` (incompletos) antes de reintentar:
  ```bash
  ASSET_ID=$(gh api repos/OWNER/REPO/releases/ID/assets --jq '.[] | select(.name=="file.gz") | .id')
  gh api -X DELETE repos/OWNER/REPO/releases/assets/$ASSET_ID
  ```

#### B. rsync (mejor para archivos <30MB o con resume)

```bash
# Con resume automatico si se corta
rsync -avz --progress -e "ssh -T -o ConnectTimeout=60 -o ServerAliveInterval=5" \
  vast:/workspace/archivo.pt ./destino/
```

#### C. Comprimir + SCP (archivos medianos)

```bash
# En remoto
ssh vast "gzip -k /workspace/archivo.pt"

# Descargar
scp -O vast:/workspace/archivo.pt.gz ./destino/

# Descomprimir
gzip -d ./destino/archivo.pt.gz
```

#### D. Split + SCP (ultimo recurso, chunks de 4MB)

```bash
# En remoto: dividir
ssh vast "split -b 4M /workspace/archivo.pt /tmp/chunk_"

# Descargar cada parte
scp -O vast:/tmp/chunk_* ./destino/

# Recombinar
cat ./destino/chunk_* > ./destino/archivo.pt

# Limpiar
ssh vast "rm /tmp/chunk_*"
```

### Opciones SSH para estabilidad

```
-T                          Sin pseudo-terminal (CRITICO para transferencias)
-o ConnectTimeout=60        Timeout de conexion generoso
-o ServerAliveInterval=30   Keepalive cada 30s
-o ServerAliveCountMax=10   Max keepalives sin respuesta
```

---

## 7. Almacenamiento

### Estructura de directorios

```
/                   -> Container overlay (efimero, se pierde al destruir)
/workspace/         -> Persistente entre reinicios (AQUI VA TODO)
  ├── flux-lm/      -> Repositorio del proyecto
  ├── corpus_*.txt   -> Corpus de entrenamiento
  └── *.log          -> Logs de entrenamiento
```

### Volumes (almacenamiento persistente externo)

```bash
# Crear volume
vastai create volume <offer_id> -s <size_GB> -n <nombre>

# Montar al crear instancia
vastai create instance <id> --image <img> -v V.<vol_id>:/mnt

# Cloud sync soportado: S3, Google Drive, Backblaze, Dropbox, HuggingFace
# NO soporta transferencia directa local -> volume
```

### Gestion de disco

```bash
# Ver espacio libre
ssh vast "df -h /workspace"

# Ver uso por directorio
ssh vast "du -sh /workspace/*"

# Checkpoints tipicos:
#   3.5M params: ~41 MB por checkpoint
#   7M params:   ~82 MB por checkpoint
# Planificar: N_checkpoints * tamanio < disco_libre
```

---

## 8. Variables de Entorno

### CRITICO: Variables custom NO son visibles en SSH

Las variables pasadas via `-e` al crear instancia NO aparecen en sesiones SSH/tmux/Jupyter.

**Solucion**: Exportarlas en onstart o escribirlas en `/etc/bash.bashrc`:

```bash
# En onstart:
echo 'export TORCH_CUDA_ARCH_LIST="8.9"' >> /etc/bash.bashrc

# O al conectar:
ssh vast 'echo "export TORCH_CUDA_ARCH_LIST=8.9" >> ~/.bashrc'
```

### Variables predefinidas por Vast.ai

| Variable | Descripcion |
|----------|-------------|
| `CONTAINER_API_KEY` | API key del container |
| `CONTAINER_ID` | ID del container |
| `GPU_COUNT` | Numero de GPUs |
| `PUBLIC_IPADDR` | IP publica |
| `SSH_PUBLIC_KEY` | Clave SSH inyectada |
| `JUPYTER_TOKEN` | Token para Jupyter |
| `PYTORCH_VERSION` | Version de PyTorch |

### Variables importantes para training

```bash
# Compilacion CUDA nativa para RTX 4090
export TORCH_CUDA_ARCH_LIST="8.9"

# CUDA paths (ya configurados en NGC)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

---

## 9. Flujo de Trabajo Tipico

### Setup inicial de instancia

```bash
# 1. Conectar
ssh vast

# 2. Verificar GPU
nvidia-smi

# 3. Clonar/subir codigo
cd /workspace
git clone https://github.com/OWNER/repo.git
# O subir con SCP (archivos pequenos):
# scp -O vast:/workspace/ archivo.py

# 4. Configurar entorno
export TORCH_CUDA_ARCH_LIST="8.9"
pip install -r requirements.txt  # si es necesario

# 5. Subir corpus (si no esta)
# Usar metodo de transferencia segun tamanio (ver seccion 6)

# 6. Verificar setup
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Lanzar entrenamiento

```bash
# Opcion A: En primer plano (con tmux para persistencia)
python train.py --corpus /workspace/corpus.txt \
  --d 1024 --layers 12 --epochs 67 \
  --batch-size 64 --seq-len 256 \
  --dtype bf16 --parallel --grad-checkpoint \
  --ckpt-every 5 --ckpt-dir checkpoints/ \
  2>&1 | tee /workspace/train.log

# Opcion B: En background con nohup
nohup python train.py [args] > /workspace/train.log 2>&1 &
echo $!  # guardar PID

# Desconectar SSH sin matar el proceso
# (tmux: Ctrl+B, D)
# (nohup: ya corre en background)
```

### Monitorear entrenamiento (desde local)

```bash
# Log en tiempo real
ssh vast "tail -f /workspace/train.log"

# Ultimo epoch
ssh vast "grep 'epoch' /workspace/train.log | tail -5"

# GPU usage
ssh vast "nvidia-smi"

# Proceso activo
ssh vast "ps aux | grep python"

# Checkpoints guardados
ssh vast "ls -lh /workspace/checkpoints/"
```

### Descargar resultados

```bash
# Ver checkpoints disponibles
ssh vast "ls -lh /workspace/checkpoints/"

# Descargar via GitHub Release (ver seccion 6.A)
# O via rsync para archivos pequenos
rsync -avz vast:/workspace/train.log ./logs/
```

---

## 10. Situaciones Comunes y Soluciones

### "Connection timed out" al conectar por SSH

1. **Verificar VPN**: Vast.ai NO funciona con VPN activa. Desactivar VPN.
2. **IP/puerto cambiaron**: Usar `vastai show instances` o consola web para obtener datos nuevos.
3. **Instancia apagada/destruida**: Verificar en consola web.

### "Connection closed by remote host" durante transferencia

- La instancia cierra conexiones SSH largas (>5 min de transferencia continua)
- Solucion: Usar GitHub Releases como intermediario o split + SCP

### Proceso de entrenamiento murio sin motivo aparente

```bash
# Verificar si fue OOM (Out of Memory)
ssh vast "dmesg | grep -i 'oom\|killed' | tail -10"

# Verificar si la instancia se reinicio
ssh vast "uptime"

# Verificar logs del proceso
ssh vast "tail -50 /workspace/train.log"
```

**Causas comunes:**
- OOM del sistema (no GPU, sino RAM del host) — reducir num_workers o batch_size
- Instancia interrumpida (tipo interruptible)
- Timeout de inactividad (si no hay keepalive SSH)
- DataLoader workers muertos por SIGTERM

### GPU no aparece o CUDA error

```bash
# Verificar driver
ssh vast "nvidia-smi"

# Verificar CUDA
ssh vast "nvcc --version"

# Verificar PyTorch ve la GPU
ssh vast "python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"

# Si CUDA falla, verificar compat library
ssh vast "ls /usr/local/cuda/compat/"
```

### Disco lleno

```bash
# Ver uso
ssh vast "df -h /workspace"
ssh vast "du -sh /workspace/* | sort -h"

# Limpiar caches de compilacion CUDA
ssh vast "rm -rf /workspace/flux-lm/torch_port/.kernel_cache/"
ssh vast "rm -rf /workspace/flux-lm/torch_port/.selective_scan_cache/"

# Eliminar checkpoints viejos (conservar el mejor)
ssh vast "rm /workspace/checkpoints/flux_epoch_00{05,10,15,20}.pt"

# Limpiar __pycache__
ssh vast "find /workspace -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null"
```

### Kernels CUDA no compilan

```bash
# Verificar nvcc
ssh vast "nvcc --version"

# Verificar arch list
ssh vast "python -c 'import torch; print(torch.cuda.get_arch_list())'"

# Forzar arch para RTX 4090
ssh vast "export TORCH_CUDA_ARCH_LIST=8.9"

# Limpiar cache y recompilar
ssh vast "rm -rf /workspace/flux-lm/torch_port/.kernel_cache/"
```

---

## 11. Networking y Puertos

- Instancias comparten IP publica; aislamiento por puertos
- Puertos internos se mapean a puertos externos **aleatorios**
- Maximo 64 puertos abiertos por instancia
- Puertos 70000+ para identity mapping (interno = externo)

```bash
# Exponer puerto al crear instancia
vastai create instance <id> --image <img> -e "-p 8080:8080"

# Port forwarding via SSH (mas seguro)
ssh vast -L 6006:localhost:6006  # TensorBoard
ssh vast -L 8888:localhost:8888  # Jupyter
```

---

## 12. Costos y Optimizacion

### Estimacion de costos (RTX 4090, sept 2026)

| Tipo | Precio/hora | 24h | 1 semana |
|------|-------------|-----|----------|
| On-demand | ~$0.40-0.60 | ~$12 | ~$84 |
| Reserved | ~$0.20-0.35 | ~$7 | ~$49 |
| Interruptible | ~$0.15-0.25 | ~$4 | ~$28 |

### Tips para reducir costos

1. **Usar reserved** para training >24h
2. **Checkpoint frecuente** (cada 5 epochs) — permite reanudar si se pierde instancia
3. **Pausar instancia** cuando no se usa (no destruir, para mantener datos)
4. **Descargar checkpoints** importantes a local/GitHub como backup
5. **Gradient checkpointing** para usar menos VRAM y poder usar GPU mas barata
6. **Comprimir datos** antes de transferir para ahorrar tiempo (y costo) de instancia activa

---

## 13. Checklist Pre-Entrenamiento

```
[ ] GPU visible: nvidia-smi muestra RTX 4090
[ ] CUDA funcional: nvcc --version
[ ] PyTorch ve GPU: torch.cuda.is_available()
[ ] TORCH_CUDA_ARCH_LIST=8.9 configurado
[ ] Codigo actualizado en /workspace/
[ ] Corpus subido y verificado (md5sum)
[ ] __pycache__ limpiado (si se actualizo codigo)
[ ] Cache de kernels limpiado (si se modifico codigo CUDA)
[ ] Forward+backward test pasa (sin OOM)
[ ] Disco suficiente para N checkpoints
[ ] Log configurado (tee o redireccion)
[ ] Checkpoints frecuentes habilitados (--ckpt-every)
[ ] VPN desactivada
```

---

## 14. Comandos de Referencia Rapida

```bash
# === INSTANCIA ===
vastai show instances              # Ver instancias activas
vastai start instance <ID>         # Iniciar instancia pausada
vastai stop instance <ID>          # Pausar instancia
vastai destroy instance <ID>       # Destruir instancia (pierde datos!)

# === SSH ===
ssh vast                           # Conectar
ssh vast "comando"                 # Ejecutar comando remoto
ssh -o ConnectTimeout=10 vast      # Con timeout

# === MONITOREO ===
ssh vast "nvidia-smi"              # GPU status
ssh vast "ps aux | grep python"    # Procesos
ssh vast "tail -20 /workspace/*.log" # Logs
ssh vast "df -h /workspace"        # Disco
ssh vast "free -h"                 # RAM

# === TRANSFERENCIA ===
scp -O vast:/remoto/file ./local/  # Descargar (<30MB)
scp -O ./local/file vast:/remoto/  # Subir (<30MB)
rsync -avz vast:/remoto/ ./local/  # Sync con resume

# === TRAINING ===
ssh vast "kill -9 <PID>"           # Matar proceso
ssh vast "nvidia-smi | grep python" # Ver si GPU en uso
```

---

## 15. Seguridad

- **Nunca** exponer tokens/API keys en comandos SSH (visibles en `ps aux`)
- **Rotar tokens** despues de usarlos en transferencias
- Configurar API key de vast localmente (`~/.vast_api_key`), no en el repo
- Usar `.gitignore` para checkpoints y datos de entrenamiento
- SSH keys: permisos `chmod 600` obligatorios en la clave privada
- Si el repo es publico temporalmente, regresarlo a privado despues de transferir

---

## 16. Multi-GPU (DDP)

### Seleccion de instancia multi-GPU

```bash
# Buscar 4x RTX 4090
vastai search offers 'gpu_name=RTX 4090 num_gpus=4 gpu_ram>=24000 disk_space>=100 reliability>0.95' \
  --order 'dph_total'
```

### Lanzar entrenamiento DDP

```bash
# Reemplazar "python3" por "torchrun --nproc_per_node=N"
torchrun --nproc_per_node=4 train.py \
    --corpus /workspace/corpus_python_200mb.txt \
    --d 1024 --layers 12 --epochs 72 \
    --batch-size 64 --seq-len 256 --lr 2e-4 \
    --dtype bf16 --parallel --n-corrections 1 \
    --grad-checkpoint --schedule cosine \
    --ckpt-every 1 --print-every 1 --num-workers 2
```

### Consideraciones DDP

| GPUs | Batch efectivo | Ajuste LR | Speedup esperado |
|------|---------------|-----------|-------------------|
| 1 | 64 | base (1e-4 o 3e-4) | 1× |
| 2 | 128 | igual | ~1.95× |
| 4 | 256 | ×2 (linear scaling) | ~3.8× |
| 8 | 512 | ×2-3 + warmup | ~7× |

- Con 2 GPUs no se necesita ajustar LR
- Con 4+ GPUs: aplicar linear scaling rule (`lr *= N_gpus / base_gpus`) y considerar warmup de 1-2 epochs
- El modelo (7M params) es pequenio — overhead de comunicacion DDP es despreciable
- train.py ya soporta DDP nativo via `torchrun`

### Costes estimados multi-GPU (RTX 4090, sept 2026)

| Config | $/hora | 5 epochs (200MB) | 30 epochs (200MB) |
|--------|--------|-------------------|---------------------|
| 1× 4090 | ~$0.45 | ~$6 (13h) | ~$35 (78h) |
| 2× 4090 | ~$0.90 | ~$6 (6.6h) | ~$35 (39h) |
| 4× 4090 | ~$2.00 | ~$7 (3.3h) | ~$40 (20h) |

Multi-GPU no ahorra dinero, ahorra tiempo. Coste total similar.

---

## 17. Resume Training (reanudacion)

### Reanudacion basica

```bash
python3 train.py \
    --corpus /workspace/corpus_python_200mb.txt \
    --d 1024 --layers 12 --epochs 72 \
    --batch-size 64 --seq-len 256 --lr 1e-4 \
    --dtype bf16 --parallel --n-corrections 1 \
    --grad-checkpoint --schedule cosine \
    --ckpt /workspace/checkpoints_7m/flux_epoch_0067.pt \
    --resume \
    --ckpt-every 1 --ckpt-dir checkpoints_7m_200mb \
    --print-every 1 --num-workers 2
```

### Que hace --resume

- Restaura `model_state`, `optimizer_state`, `epoch` y `loss` del checkpoint
- El schedule `cosine` calcula: `progress = (epoch - start_epoch) / (total_epochs - start_epoch)`
  - Con `--epochs 72` y checkpoint epoch 67: crea cosine fresco de lr=1.0 a lr=0 en 5 epochs
- `sign_history` (int64) del EntropicAdam se restaura correctamente (backup/restore para evitar corrupcion float32)

### Cambiar corpus al reanudar

Es valido reanudar con un corpus distinto (mas grande). El modelo conserva los pesos aprendidos. Efectos:
- test_bpb subira temporalmente (distribucion ligeramente distinta)
- Convergencia mas rapida que entrenar desde cero
- El floor de convergencia deberia ser inferior gracias a mayor diversidad

### Compatibilidad checkpoint/codigo

> **CRITICO**: ver seccion de compatibilidad en `VAST_SETUP.md`. Si el codigo cambio entre el checkpoint y el codigo actual, pueden haber params faltantes que se inicializan incorrectamente.

---

## 18. Configuraciones de Entrenamiento Probadas

### Run exitosos (resultados verificados)

| Run | Modelo | Corpus | Epochs | LR | BPB final | Checkpoint |
|-----|--------|--------|--------|-----|-----------|------------|
| v1-v5 | 3.5M (d=512) | 60MB | 50 | 3e-4 | **0.963** | flux_epoch_0050.pt |
| v6 | 7M (d=1024) | 60MB | 67 | 3e-4 | 1.092 | flux_epoch_0067.pt |
| v7 | 7M (d=1024) | 200MB | 68-72 | 1e-4 | (en curso) | checkpoints_7m_200mb/ |

### Run fallidos (evitar repetir)

| Run | Error | Causa |
|-----|-------|-------|
| v4 (SGDR) | test_loss degrado 0.668→0.840 | Weight decay no escalaba con lr_scale |
| v5 (SGDR ciclos 2-3) | Nunca recupero minimo | LR pico 3e-4 demasiado alto post-convergencia |

### Flags que SIEMPRE se deben usar

```
--parallel            # Parallel scan (4-7× mas rapido que sequential)
--grad-checkpoint     # Reduce VRAM ~81% (critico para batch-size 64)
--n-corrections 1     # Correccion perturbativa (mejora calidad)
--dtype bf16          # Mixed precision (2× throughput)
--num-workers 2       # DataLoader paralelo
```
