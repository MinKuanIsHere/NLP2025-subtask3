FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        faiss-cpu==1.8.0.post1 \
        faiss-gpu-cu12==1.8.0.2

COPY app.py .

CMD ["/bin/bash"]
