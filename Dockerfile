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
# ONNX Runtime GPU : wheel de ~291 Mo. On l'installe depuis .build-cache/ si
# présent (téléchargement manuel), sinon depuis PyPI avec des retries explicites.
RUN --mount=type=bind,source=.build-cache,target=/tmp/build-cache \
    if [ -f /tmp/build-cache/onnxruntime_gpu-*.whl ]; then \
        pip install --no-cache-dir /tmp/build-cache/onnxruntime_gpu-*.whl; \
    else \
        pip install --no-cache-dir --timeout 120 --retries 10 onnxruntime-gpu==1.20.2; \
    fi

RUN pip install --no-cache-dir --timeout 120 --retries 10 \
        numpy==1.26.4 \
        onnx==1.17.0 \
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

# TensorRT : les bibliothèques pèsent ~2 Go. Si le wheel est présent dans
# .build-cache/, on l'installe localement (utile quand pypi.nvidia.com est lent) ;
# sinon on laisse pip télécharger le paquet officiel. Le bind mount évite de
# conserver le wheel de 2 Go dans les couches de l'image.
RUN --mount=type=bind,source=.build-cache,target=/tmp/build-cache \
    if [ -f /tmp/build-cache/tensorrt_cu12_libs-10.7.0.post1-py2.py3-none-manylinux_2_17_x86_64.whl ]; then \
        pip install --no-cache-dir --no-deps /tmp/build-cache/tensorrt_cu12_libs-*.whl; \
        pip install --no-cache-dir --no-deps /tmp/build-cache/tensorrt_cu12_bindings-*.whl \
            || pip install --no-cache-dir --no-deps tensorrt-cu12-bindings==10.7.0.post1; \
    else \
        pip install --no-cache-dir --timeout 120 --retries 10 tensorrt==10.7.0.post1; \
    fi && \
    rm -f /opt/conda/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_builder_resource_win.so.10.7.0
# Le module Python `tensorrt` est fourni par le sdist tensorrt-cu12 ; on le
# recrée simplement au-dessus des bindings pour éviter le pip-inside-pip.
RUN mkdir -p /opt/conda/lib/python3.11/site-packages/tensorrt \
    && printf 'from tensorrt_bindings import *\nfrom tensorrt_bindings import __version__\n' \
       > /opt/conda/lib/python3.11/site-packages/tensorrt/__init__.py

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
# Certains composants (torch.compile/Inductor, Triton) appellent getpwuid().
# L'UID 1000 doit donc exister dans /etc/passwd, même si l'entrypoint drop
# vers cet UID.
RUN if ! getent group 1000 >/dev/null; then groupadd -g 1000 tp; fi \
    && if ! getent passwd 1000 >/dev/null; then useradd -u 1000 -g 1000 -m -s /bin/bash tp; fi

COPY docker/tp-entrypoint.py /usr/local/bin/tp-entrypoint
COPY docker/tp-rapl-daemon.py /usr/local/bin/tp-rapl-daemon
RUN chmod +x /usr/local/bin/tp-entrypoint /usr/local/bin/tp-rapl-daemon

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/tp-entrypoint"]
CMD ["bash"]
