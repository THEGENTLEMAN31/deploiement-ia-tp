#!/usr/bin/env bash
set -euo pipefail

cd /workspace

# Exécute chaque notebook en place (les sorties sont conservées pour les captures
# d'écran et la correction). Les notebooks écrivent leurs métriques dans results/.
NOTEBOOKS=(
  "notebooks/00_reference_pytorch.ipynb"
  "notebooks/01_cpu_onnx_openvino.ipynb"
  "notebooks/02_cpu_torchscript_compile.ipynb"
  "notebooks/03_gpu_onnx_tensorrt.ipynb"
  "notebooks/04_gpu_torch_compile.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
  echo "============================================================"
  echo "Exécution : ${nb}"
  echo "============================================================"
  jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    "${nb}"
done

echo "Tous les notebooks ont été exécutés."
