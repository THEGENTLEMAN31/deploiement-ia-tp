#!/usr/bin/env bash
# Pipeline complet, à exécuter DANS le conteneur :
#   docker compose run --rm tp bash scripts/run_all.sh
set -euo pipefail
cd /workspace
mkdir -p report/logs

nvidia-smi > report/logs/nvidia-smi.txt 2>&1 || true
python - <<'PY' > report/logs/torch-env.txt 2>&1
import torch, platform
print("Python :", platform.python_version())
print("PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("CUDA :", torch.version.cuda)
print("cuDNN :", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
    print("Mémoire GPU :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "Go")
PY

python -m src.prepare 2>&1 | tee report/logs/prepare.txt
bash scripts/execute_notebooks.sh 2>&1 | tee report/logs/notebooks.txt
python -m src.make_plots 2>&1 | tee report/logs/plots.txt
python -m src.make_report 2>&1 | tee report/logs/report.txt
python scripts/make_notebook_screenshots.py 2>&1 | tee report/logs/notebook-html.txt
python scripts/make_terminal_html.py 2>&1 | tee report/logs/terminal.txt

echo
echo "Terminé. Rapport HTML : report/rapport.html"
echo "Pages notebook prêtes pour capture : report/notebook_html/"
