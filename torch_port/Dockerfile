FROM nvcr.io/nvidia/pytorch:26.08-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.9"

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Single GPU
# docker run --gpus all -v /data:/data flux-lm \
#   python3 train.py --corpus /data/corpus.txt --d 512 --layers 12

# Multi-GPU DDP
# docker run --gpus all -v /data:/data flux-lm \
#   torchrun --nproc_per_node=4 train.py --corpus /data/corpus.txt --d 512 --layers 12

ENTRYPOINT ["python3"]
CMD ["train.py", "--help"]
