# TP Déploiement IA — Quatre pipelines CPU et GPU

Comparaison expérimentale de quatre chaînes de déploiement d'un modèle
**ResNet-18** sur **CIFAR-10** :

| # | Pipeline | Device | Format | Runtime / compilateur | Bas niveau |
|---|---|---|---|---|---|
| CPU 1 | PyTorch → ONNX → OpenVINO | CPU | ONNX | OpenVINO Runtime | oneDNN / kernels CPU |
| CPU 2 | PyTorch → TorchScript / torch.compile | CPU | TorchScript / graphe PyTorch 2 | Inductor | kernels C++/OpenMP |
| GPU 1 | PyTorch → ONNX → TensorRT | GPU | ONNX | TensorRT (FP16) | CUDA / cuDNN |
| GPU 2 | PyTorch → TorchScript / torch.compile | GPU | TorchScript / graphe PyTorch 2 | Inductor + Triton | CUDA / kernels Triton |

Le dépôt contient les notebooks, le code de préparation des données, les mesures
JSON/CSV, les figures, les captures d'écran et le rapport technique PDF.

---

## 1. Objectif

Le TP demande de comprendre l'impact des couches logicielles d'une chaîne de
déploiement sur les performances d'inférence. Pour chaque pipeline, nous mesurons :

- le **débit** (images/s) et la **latence** (ms/image) ;
- la **Top-1** sur 50 images CIFAR-10 ;
- l'**accord des prédictions** avec PyTorch eager ;
- l'**énergie** CPU (RAPL) et GPU (NVML), lorsque disponible ;
- la **stabilité** : 5 répétitions indépendantes, moyenne, écart-type, min, max.

Les quatre pipelines utilisent **exactement le même modèle, les mêmes poids, les
mêmes 50 images, le même prétraitement et le même batch**.

---

## 2. Point méthodologique important : 1000 classes ImageNet vs 10 classes CIFAR-10

ResNet-18 est pré-entraîné sur ImageNet (1000 classes), alors que le TP impose
CIFAR-10 (10 classes). Utiliser directement la sortie 1000 classes ne permet pas
de calculer une Top-1 sur 10 classes.

Nous avons donc adopté un **linear probe** :

1. backbone ResNet-18 ImageNet **gelé** ;
2. remplacement de `fc` par une couche linéaire `512 → 10` ;
3. entraînement de cette tête sur les 50 000 images d'entraînement CIFAR-10
   (features extraites une fois), avec 5 000 images de validation ;
4. sauvegarde des poids dans `artifacts/classifier_head.npz`, rechargés à
   l'identique par les quatre pipelines.

Cette adaptation est documentée dans le rapport. Elle ne biaise pas la comparaison
des pipelines : le graphe et les poids sont identiques, seules les couches
d'optimisation changent.

Accuracy obtenue (à compléter après exécution) :

- test complet CIFAR-10 (10 000 images) : **voir `artifacts/prepare_metadata.json`** ;
- sous-ensemble de 50 images : **voir `results/00_reference_pytorch_cpu.json`**.

---

## 3. Organisation du dépôt

```
.
├── Dockerfile                     # image unique CPU + GPU
├── docker-compose.yml             # service "tp", GPU, RAPL, UID hôte
├── Makefile                       # build / prepare / notebooks / report
├── README.md
├── artifacts/
│   ├── cifar10_subset.npz         # 50 images de test (5 par classe)
│   ├── classifier_head.npz        # tête linéaire 10 classes
│   └── prepare_metadata.json      # protocole, accuracies, environnement
├── notebooks/
│   ├── 00_reference_pytorch.ipynb # référence eager CPU + GPU
│   ├── 01_cpu_onnx_openvino.ipynb
│   ├── 02_cpu_torchscript_compile.ipynb
│   ├── 03_gpu_onnx_tensorrt.ipynb
│   └── 04_gpu_torch_compile.ipynb
├── src/
│   ├── common.py                 # données, modèle, benchmark, énergie, IO
│   ├── prepare.py                # CIFAR-10 + linear probe
│   ├── make_plots.py             # figures comparatives
│   └── make_report.py            # rapport HTML
├── results/                       # JSON, CSV, logits
├── report/
│   ├── rapport.html
│   ├── rapport.pdf
│   ├── figures/
│   └── screenshots/
├── scripts/
│   ├── make_notebooks.py          # régénère les .ipynb
│   └── execute_notebooks.sh       # exécute les 5 notebooks
└── docker/
    ├── tp-entrypoint.py           # drop de privilèges + démon RAPL
    └── tp-rapl-daemon.py          # lecture RAPL pour les kernels Jupyter
```

---

## 4. Démarrage rapide (Docker)

Prérequis : Docker, Docker Compose, GPU NVIDIA + NVIDIA Container Toolkit pour
les notebooks 03 et 04.

```bash
# 1. Construire l'image (PyTorch CUDA 12.4, ONNX, OpenVINO, TensorRT, Jupyter)
docker compose build

# 2. Télécharger CIFAR-10, sélectionner les 50 images, entraîner le linear probe
docker compose run --rm tp python -m src.prepare

# 3. Exécuter les 5 notebooks (référence + 4 pipelines)
docker compose run --rm tp bash scripts/execute_notebooks.sh

# 4. Générer les figures et le rapport HTML
docker compose run --rm tp python -m src.make_plots
docker compose run --rm tp python -m src.make_report
```

Le PDF est ensuite produit en imprimant `report/rapport.html` avec Chromium
(voir section 6).

### Pourquoi `privileged: true` ?

Le service est privilégié **uniquement** pour exposer `/sys/class/powercap`
(compteurs RAPL d'énergie CPU). L'entrypoint ouvre les compteurs en root, démarre
un petit démon RAPL, puis **abandonne les privilèges** vers l'UID 1000 avant
d'exécuter la commande. Les fichiers créés dans le dépôt restent donc possédés par
l'utilisateur hôte. On peut retirer `privileged: true` : les benchmarks fonctionnent
toujours, seule l'énergie CPU devient indisponible.

### Sans GPU

Les notebooks 00, 01 et 02 fonctionnent sur CPU. Les notebooks 03 et 04 nécessitent
un GPU NVIDIA. Sur une machine sans GPU, on peut exécuter uniquement les pipelines
CPU et adapter le rapport.

---

## 5. Google Colab

Les notebooks sont compatibles Colab. Pour les notebooks 03 et 04, choisir
**Exécution → Modifier le type d'exécution → GPU**.

Dans Colab, le dépôt n'est pas présent par défaut. Définir l'URL du dépôt dans la
variable d'environnement `TP_REPO_URL` avant d'exécuter la cellule de setup :

```python
import os
os.environ["TP_REPO_URL"] = "https://github.com/<votre-compte>/deploiement-ia-tp.git"
```

La cellule de setup clone le dépôt si nécessaire, se place à sa racine et importe
`src/common.py`. Les dépendances sont déjà présentes dans l'image Docker ; sur
Colab, installer si besoin :

```bash
pip install onnx onnxruntime-gpu openvino tensorrt pandas matplotlib seaborn
```

---

## 6. Rapport PDF

Le rapport est généré en HTML par `src/make_report.py`, puis imprimé en PDF. Deux
options :

### Option A — Chromium / Playwright (recommandé)

```bash
# Si Playwright est installé sur l'hôte
python - <<'PY'
from playwright.sync_api import sync_playwright
from pathlib import Path
html = Path("report/rapport.html").resolve().as_uri()
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(html, wait_until="networkidle")
    page.pdf(path="report/rapport.pdf", format="A4",
             print_background=True, margin={"top":"17mm","bottom":"17mm","left":"15mm","right":"15mm"})
    browser.close()
PY
```

### Option B — impression navigateur

Ouvrir `report/rapport.html` dans Chrome/Chromium, puis
**Imprimer → Enregistrer au format PDF → A4 → marges par défaut**.

---

## 7. Protocole expérimental

| Paramètre | Valeur |
|---|---|
| Modèle | ResNet-18 + tête linéaire 10 classes |
| Poids | ImageNet `IMAGENET1K_V1` + linear probe CIFAR-10 |
| Données | 50 images de test CIFAR-10, 5 par classe |
| Prétraitement | 32×32 → 224×224 bilinéaire antialias, normalisation ImageNet |
| Batch | 10 images (5 batchs pour 50 images) |
| Warmups | 5 passes complètes |
| Répétitions | 5 exécutions indépendantes |
| Débit | `50 / temps_total` |
| Latence | `temps_total / 50` |
| Statistiques | moyenne, écart-type (échantillon), min, max |
| Énergie CPU | RAPL `package-0` (µJ → J), via démon Docker |
| Énergie GPU | NVML `nvmlDeviceGetTotalEnergyConsumption` (mJ → J), fallback puissance |
| Seed | 42 |

Le prétraitement est effectué **avant** la boucle chronométrée : on mesure
l'inférence, pas le décodage ni le redimensionnement.

---

## 8. Résultats

Les résultats complets sont dans `results/` et `report/rapport.pdf`. Les figures
principales sont :

- `report/figures/fig_throughput.png` — débit CPU et GPU ;
- `report/figures/fig_latency.png` — latence ;
- `report/figures/fig_accuracy.png` — Top-1 ;
- `report/figures/fig_energy.png` — énergie par image ;
- `report/figures/fig_variability.png` — variabilité sur 5 répétitions ;
- `report/figures/fig_speedup.png` — gain vs PyTorch eager ;
- `report/figures/fig_agreement.png` — accord des prédictions.

**À compléter après exécution** : reprendre les valeurs de `results/summary.csv`
et le résumé du rapport.

---

## 9. Reproductibilité

- Toutes les mesures sont produites par les notebooks, avec une seed fixe.
- Les JSON contiennent l'environnement complet (versions, CPU, GPU, driver).
- Les notebooks sont exécutés avec `nbconvert --execute`, donc les sorties sont
  réellement celles du conteneur.
- Les figures et le rapport sont régénérés à partir des JSON, sans recopie manuelle.
- Les artefacts lourds (ONNX, moteurs TensorRT) sont regénérables et ignorés par
  Git ; les 50 images et la tête linéaire (quelques centaines de Ko) sont versionnées.

---

## 10. Limites

- Les mesures locales (Intel Core i5-10200H, RTX 3050 Laptop 4 Go) ne sont pas
  directement transposables à Google Colab ; les tendances restent valables.
- 50 images est un petit échantillon : la Top-1 varie de plusieurs points selon
  un ou deux exemples. L'accord de prédictions est plus robuste.
- Le linear probe est une adaptation nécessaire à l'incohérence 1000/10 classes.
- L'énergie CPU RAPL est celle du package, pas du seul modèle.
- TensorRT est utilisé via l'Execution Provider d'ONNX Runtime, pas via `trtexec`.

---

## 11. Licence

Usage pédagogique. Le code est fourni tel quel, sans garantie.
