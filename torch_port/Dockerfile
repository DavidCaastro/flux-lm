FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

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
