# Image unique pour les 4 pipelines (CPU + GPU).
# Base PyTorch officielle CUDA 12.4 / cuDNN 9, compatible avec le driver de l'hôte.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# torchvision doit venir de l'index CUDA pour rester aligné avec torch 2.5.1+cu124.
RUN pip install --no-cache-dir \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Dépendances des pipelines et du rapport.
# - onnx / onnxruntime-gpu : pipeline ONNX (CPU et GPU)
# - tensorrt              : moteur TensorRT (via l'Execution Provider d'ONNX Runtime)
# - openvino              : runtime CPU Intel
# - jupyter/nbconvert     : exécution et export des notebooks
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        onnx==1.17.0 \
        onnxruntime-gpu==1.20.2 \
        tensorrt==10.7.0.post1 \
        openvino==2025.1.0 \
        pandas==2.2.3 \
        matplotlib==3.9.2 \
        seaborn==0.13.2 \
        psutil==6.1.0 \
        nvidia-ml-py==12.560.30 \
        scikit-learn==1.5.2 \
        jupyterlab==4.3.0 \
        nbconvert==7.16.4 \
        nbclient==0.10.0 \
        ipykernel==6.29.5

# torch.compile / Inductor a besoin d'un compilateur C++ et de ninja.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir ninja==1.11.1.1

# Les bibliothèques TensorRT installées par pip ne sont pas dans le chemin
# standard du loader ; on les rend visibles à ONNX Runtime (TensorRT EP).
RUN echo "/opt/conda/lib/python3.11/site-packages/tensorrt_libs" \
        > /etc/ld.so.conf.d/tensorrt.conf \
    && ldconfig

# Entrypoint : ouvre les compteurs RAPL en root, puis abandonne les privilèges
# pour que tous les fichiers du workspace restent possédés par l'utilisateur hôte.
COPY docker/tp-entrypoint.py /usr/local/bin/tp-entrypoint
COPY docker/tp-rapl-daemon.py /usr/local/bin/tp-rapl-daemon
RUN chmod +x /usr/local/bin/tp-entrypoint /usr/local/bin/tp-rapl-daemon

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/tp-entrypoint"]
CMD ["bash"]
