#!/usr/bin/env python3
"""Génère les cinq notebooks du TP (format .ipynb).

Les notebooks sont volontairement générés par script pour garantir que les
cellules de protocole (setup, données, affichage) sont identiques partout.
Exécution : python scripts/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NB_DIR / filename).write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print("écrit :", NB_DIR / filename)


SETUP = r'''
%matplotlib inline
import json
import os
import sys
import time
from pathlib import Path

def _find_root() -> Path:
    p = Path.cwd().resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "src" / "common.py").exists():
            return candidate
    return p

ROOT = _find_root()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TP_ROOT", str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

IN_COLAB = "google.colab" in sys.modules
REPO_URL = os.environ.get("TP_REPO_URL", "")
if IN_COLAB and REPO_URL and not (ROOT / "src" / "common.py").exists():
    import subprocess
    subprocess.run(["git", "clone", REPO_URL], check=True)
    ROOT = Path(REPO_URL.rstrip("/").split("/")[-1].removesuffix(".git")).resolve()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ["TP_ROOT"] = str(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Markdown
from src import common as C

C.ensure_dirs()
C.set_seed()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
print("Racine du projet :", ROOT)
print("PyTorch :", torch.__version__, "| CUDA :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
print(f"Protocole : {C.N_IMAGES} images | batch={C.BATCH_SIZE} | {C.REPETITIONS} répétitions | warmup={C.WARMUP}")
'''

HELPERS = r'''
def show(result_or_id):
    result = C.load_result(result_or_id) if isinstance(result_or_id, str) else result_or_id
    print(f"{result['label']}  |  device={result['device']}  |  backend={result['backend']}")
    display(pd.DataFrame(C.summary_rows(result)).round(4))
    print(f"Top-1 accuracy = {result['accuracy_top1'] * 100:.2f} %")

def bar_metric(ids, metric, std_metric, ylabel, title, color="#4C72B0"):
    df = pd.DataFrame(C.comparison_rows(ids))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["pipeline"], df[metric], yerr=df[std_metric], capsize=5, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=18)
    for i, value in enumerate(df[metric]):
        if value is not None:
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()
    return df
'''

DATA = r'''
images, labels = C.load_subset()
x_cpu = C.preprocess_images(images, device="cpu")
print("images :", images.shape, images.dtype)
print("tenseur prétraité :", tuple(x_cpu.shape), x_cpu.dtype)
print("distribution :", np.bincount(labels, minlength=len(C.CLASSES)).tolist())
print("classes :", C.CLASSES)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, idx in zip(axes.flat, range(10)):
    ax.imshow(images[idx])
    ax.set_title(C.CLASSES[int(labels[idx])])
    ax.axis("off")
fig.suptitle("Exemples du sous-ensemble CIFAR-10 (5 premières classes)")
plt.tight_layout()
plt.show()
'''

MODEL = r'''
model = C.build_model("cpu")
n_params = sum(p.numel() for p in model.parameters())
print(f"ResNet-18 + tête 10 classes : {n_params/1e6:.2f} M paramètres")
print("Couche de classification :", model.fc)
print("Poids de la tête :", C.ARTIFACTS / "classifier_head.npz")
'''

NOTEBOOK_00 = [
    md("""
# 00 — Référence PyTorch eager (CPU et GPU)

Ce notebook n'est pas l'un des quatre pipelines demandés : il établit la
**référence non optimisée**. Toutes les comparaisons de débit, de latence,
d'accuracy et d'accord de prédictions se font par rapport à cette exécution.

- Modèle : ResNet-18 ImageNet + tête linéaire 10 classes (linear probe CIFAR-10).
- Données : 50 images CIFAR-10 de test (5 par classe), redimensionnées en 224×224
  et normalisées avec les paramètres ImageNet.
- Protocole : 5 répétitions indépendantes, 5 warmups, batch de 10, 50 images.
- Mesures : débit (images/s), latence (ms/image), énergie CPU/GPU, Top-1.
"""),
    code(SETUP),
    code(HELPERS),
    md("## 1. Données et prétraitement"),
    code(DATA),
    md("## 2. Modèle commun"),
    code(MODEL),
    md("""
## 3. Référence CPU — PyTorch eager

C'est le pipeline « haut niveau → matériel » sans format intermédiaire ni
runtime optimisé : PyTorch exécute directement les opérateurs sur le CPU.
"""),
    code(r'''
ref_cpu = C.ensure_reference("cpu", x_cpu, labels)
show(ref_cpu)
'''),
    code(r'''
records = pd.DataFrame(ref_cpu["records"])
ax = records.plot(x="repetition", y="throughput_ips", marker="o", legend=False, figsize=(8, 3))
ax.set_xlabel("Répétition")
ax.set_ylabel("images/s")
ax.set_title("Stabilité du débit — PyTorch eager CPU")
ax.set_xticks(records["repetition"])
plt.tight_layout()
plt.show()
records[["repetition", "throughput_ips", "latency_ms_per_image", "energy_cpu_j"]].round(4)
'''),
    md("""
## 4. Référence GPU — PyTorch eager (si GPU disponible)

Sur Colab, sélectionnez un runtime GPU. En local, le conteneur est lancé avec
`--gpus all`.
"""),
    code(r'''
if torch.cuda.is_available():
    ref_gpu = C.ensure_reference("cuda", x_cpu, labels)
    show(ref_gpu)
else:
    ref_gpu = None
    print("GPU indisponible : la référence GPU ne peut pas être calculée.")
'''),
    md("## 5. Prédictions de la référence CPU"),
    code(r'''
logits = np.load(C.RESULTS / "00_reference_pytorch_cpu_logits.npy")
pred = logits.argmax(axis=1)
df_pred = pd.DataFrame({
    "image": np.arange(len(labels)),
    "vraie classe": [C.CLASSES[int(i)] for i in labels],
    "prédiction": [C.CLASSES[int(i)] for i in pred],
    "correct": pred == labels,
    "confiance": logits.max(axis=1),
})
display(df_pred.head(20).round(3))
print(f"Top-1 sur les 50 images : {C.top1_accuracy(logits, labels) * 100:.2f} %")
'''),
    md("""
## 6. Comparaison CPU / GPU de la référence

Les deux exécutions utilisent exactement les mêmes poids et les mêmes entrées.
L'écart de débit attendu est important : le GPU est conçu pour les opérations
matricielles massives, tandis que le CPU reste limité par ses cœurs et sa
hiérarchie mémoire.
"""),
    code(r'''
ids = ["00_reference_pytorch_cpu"]
if ref_gpu is not None:
    ids.append("00_reference_pytorch_gpu")
table = pd.DataFrame(C.comparison_rows(ids))
display(table.round(4))
bar_metric(ids, "throughput_ips", "throughput_std", "images/s",
           "Référence PyTorch eager : débit CPU vs GPU")
'''),
    md("""
### Interprétation

La référence mesure le coût des opérateurs PyTorch sans optimisation de
déploiement. Les notebooks suivants vont comparer chaque pipeline optimisé à
cette base. La variabilité (écart-type) est calculée sur les 5 répétitions :
elle permet de distinguer un vrai gain d'un simple bruit de mesure.
"""),
]


NOTEBOOK_01 = [
    md("""
# 01 — Pipeline CPU n°1 : PyTorch → ONNX → OpenVINO Runtime → CPU

| Couche | Technologie |
|---|---|
| Haut niveau | PyTorch, ResNet-18 + tête 10 classes |
| Format du modèle | ONNX (opset 17, batch fixe 10) |
| Middle level | OpenVINO Runtime (fusion d'opérateurs, sélection de kernels) |
| Bas niveau | oneDNN / kernels CPU Intel |
| Matériel | CPU |

Le notebook exporte le modèle en ONNX, mesure d'abord ONNX Runtime CPU comme
contrôle, puis convertit le même graphe ONNX pour OpenVINO Runtime.
"""),
    code(SETUP),
    code(HELPERS),
    md("## 1. Données et modèle"),
    code(DATA),
    code(MODEL),
    md("## 2. Export ONNX"),
    code(r'''
import onnx

onnx_path = C.ARTIFACTS / f"resnet18_cifar10_b{C.BATCH_SIZE}.onnx"
C.export_onnx(model, onnx_path)
print("ONNX :", onnx_path)
print(f"taille : {onnx_path.stat().st_size / 1e6:.2f} Mo")
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("opset :", onnx_model.opset_import[0].version)
print("entrées :", [i.name for i in onnx_model.graph.input])
print("sorties :", [o.name for o in onnx_model.graph.output])
print("nombre de nœuds :", len(onnx_model.graph.node))
'''),
    md("""
## 3. Contrôle : ONNX Runtime CPU

ONNX Runtime exécute le graphe ONNX avec ses propres kernels CPU. Cette mesure
intermédiaire permet d'isoler l'effet du passage ONNX → OpenVINO.
"""),
    code(r'''
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = C.CPU_THREADS
so.inter_op_num_threads = 1
sess_cpu = ort.InferenceSession(
    str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
)
print("providers :", sess_cpu.get_providers())
print("input :", sess_cpu.get_inputs()[0].name, sess_cpu.get_inputs()[0].shape)
print("output :", sess_cpu.get_outputs()[0].name, sess_cpu.get_outputs()[0].shape)

fn = C.make_onnx_fn(sess_cpu, sess_cpu.get_inputs()[0].name)
records, summary, logits = C.run_benchmark(fn, x_cpu, "cpu")
C.save_result(
    "01a_cpu_onnxruntime", records, summary, logits, labels,
    device="cpu", backend="onnxruntime-cpu",
    notes="Contrôle ONNX Runtime CPU, même graphe ONNX.",
)
show("01a_cpu_onnxruntime")
'''),
    md("""
## 4. Conversion et exécution OpenVINO

OpenVINO lit directement le fichier ONNX, applique ses optimisations
(fusion de convolutions/BN/ReLU, sélection de kernels oneDNN) et compile le
modèle pour le CPU.
"""),
    code(r'''
import openvino as ov

core = ov.Core()
ov_model = ov.convert_model(str(onnx_path))
compiled = core.compile_model(
    ov_model,
    "CPU",
    {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": str(C.CPU_THREADS)},
)
print("OpenVINO :", ov.__version__)
print("input :", compiled.input(0).shape)
print("output :", compiled.output(0).shape)

fn_ov = C.make_openvino_fn(compiled)
records, summary, logits_ov = C.run_benchmark(fn_ov, x_cpu, "cpu")
C.save_result(
    "01_cpu_onnx_openvino", records, summary, logits_ov, labels,
    device="cpu", backend="openvino",
    notes="ONNX -> OpenVINO Runtime CPU, PERFORMANCE_HINT=THROUGHPUT, 4 threads.",
)
show("01_cpu_onnx_openvino")
'''),
    md("## 5. Comparaison et intégrité des prédictions"),
    code(r'''
ref = np.load(C.RESULTS / "00_reference_pytorch_cpu_logits.npy")
for rid in ["01a_cpu_onnxruntime", "01_cpu_onnx_openvino"]:
    r = C.load_result(rid)
    logits_r = np.load(C.RESULTS / f"{rid}_logits.npy")
    print(
        f"{r['label']:<32} Top-1={r['accuracy_top1']*100:5.2f} %  "
        f"accord avec PyTorch={C.prediction_agreement(logits_r, ref)*100:5.2f} %"
    )
'''),
    code(r'''
ids = ["00_reference_pytorch_cpu", "01a_cpu_onnxruntime", "01_cpu_onnx_openvino"]
display(pd.DataFrame(C.comparison_rows(ids)).round(4))
bar_metric(ids, "throughput_ips", "throughput_std", "images/s",
           "Pipeline CPU n°1 : débit", color="#55A868")
bar_metric(ids, "latency_ms", "latency_std", "ms / image",
           "Pipeline CPU n°1 : latence", color="#C44E52")
'''),
    md("""
### Interprétation

OpenVINO doit réduire la latence et augmenter le débit par rapport à PyTorch
eager, grâce à la fusion d'opérateurs et à la sélection de kernels oneDNN.
L'accuracy Top-1 doit rester identique (ou quasiment) : les optimisations de
déploiement ne doivent pas modifier les décisions du modèle. L'accord de
prédictions avec PyTorch quantifie cette conservation.
"""),
]


NOTEBOOK_02 = [
    md("""
# 02 — Pipeline CPU n°2 : PyTorch → TorchScript / torch.compile → CPU

| Couche | Technologie |
|---|---|
| Haut niveau | PyTorch, ResNet-18 + tête 10 classes |
| Format du modèle | TorchScript (graphe figé) / graphe PyTorch 2 + Inductor |
| Middle level | Compilation et optimisation du graphe PyTorch |
| Bas niveau | Kernels CPU générés/sélectionnés par PyTorch |
| Matériel | CPU |

Ce notebook mesure deux variantes natives de PyTorch : TorchScript (tracé +
gel + optimisation pour l'inférence) et `torch.compile` (Inductor, génération
de kernels CPU).
"""),
    code(SETUP),
    code(HELPERS),
    md("## 1. Données et modèle"),
    code(DATA),
    code(MODEL),
    md("## 2. TorchScript"),
    code(r'''
example = torch.randn(C.BATCH_SIZE, 3, C.IMAGE_SIZE, C.IMAGE_SIZE)
traced = torch.jit.trace(model, example)
traced = torch.jit.freeze(traced.eval())
traced = torch.jit.optimize_for_inference(traced)
ts_path = C.ARTIFACTS / "resnet18_cifar10_torchscript_cpu.pt"
torch.jit.save(traced, ts_path)
print("TorchScript :", ts_path, f"({ts_path.stat().st_size/1e6:.2f} Mo)")

fn_ts = C.make_torch_fn(traced, "cpu")
records, summary, logits_ts = C.run_benchmark(fn_ts, x_cpu, "cpu")
C.save_result(
    "02a_cpu_torchscript", records, summary, logits_ts, labels,
    device="cpu", backend="torchscript",
    notes="TorchScript tracé, gelé et optimisé pour l'inférence.",
)
show("02a_cpu_torchscript")
'''),
    md("""
## 3. torch.compile (Inductor)

`torch.compile` capture le graphe, le compile avec Inductor et génère des
kernels C++/OpenMP pour le CPU. La première exécution compile ; les warmups
absorbent ce coût avant les mesures.
"""),
    code(r'''
compiled = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
fn_compiled = C.make_torch_fn(compiled, "cpu")
records, summary, logits_compiled = C.run_benchmark(fn_compiled, x_cpu, "cpu")
C.save_result(
    "02_cpu_torchscript_compile", records, summary, logits_compiled, labels,
    device="cpu", backend="torch.compile/inductor",
    notes="torch.compile mode=default, Inductor CPU, 4 threads.",
)
show("02_cpu_torchscript_compile")
'''),
    md("## 4. Comparaison"),
    code(r'''
ref = np.load(C.RESULTS / "00_reference_pytorch_cpu_logits.npy")
for rid in ["02a_cpu_torchscript", "02_cpu_torchscript_compile"]:
    r = C.load_result(rid)
    logits_r = np.load(C.RESULTS / f"{rid}_logits.npy")
    print(
        f"{r['label']:<34} Top-1={r['accuracy_top1']*100:5.2f} %  "
        f"accord={C.prediction_agreement(logits_r, ref)*100:5.2f} %"
    )
ids = ["00_reference_pytorch_cpu", "02a_cpu_torchscript", "02_cpu_torchscript_compile"]
display(pd.DataFrame(C.comparison_rows(ids)).round(4))
bar_metric(ids, "throughput_ips", "throughput_std", "images/s",
           "Pipeline CPU n°2 : débit", color="#8172B2")
bar_metric(ids, "latency_ms", "latency_std", "ms / image",
           "Pipeline CPU n°2 : latence", color="#CCB974")
'''),
    md("""
### Interprétation

TorchScript fige le graphe et permet des optimisations statiques, tandis que
`torch.compile` va plus loin en générant des kernels spécialisés. Le gain dépend
fortement du CPU, du nombre de threads et du coût de compilation. Les warmups
sont exclus des mesures : on mesure le régime établi, pas la compilation.
"""),
]


NOTEBOOK_03 = [
    md("""
# 03 — Pipeline GPU n°1 : PyTorch → ONNX → TensorRT → CUDA/cuDNN → GPU

| Couche | Technologie |
|---|---|
| Haut niveau | PyTorch, ResNet-18 + tête 10 classes |
| Format du modèle | ONNX (opset 17, batch fixe 10) |
| Middle level | TensorRT : fusion d'opérateurs, kernels spécialisés, FP16 |
| Bas niveau | CUDA / cuDNN |
| Matériel | GPU NVIDIA |

Le notebook exporte le modèle en ONNX, mesure ONNX Runtime CUDA comme contrôle,
puis active l'Execution Provider TensorRT d'ONNX Runtime avec FP16 et cache de
moteur.
"""),
    code(SETUP),
    code(HELPERS),
    md("## 1. Données et modèle"),
    code(DATA),
    code(MODEL),
    md("## 2. Export ONNX"),
    code(r'''
import onnx

onnx_path = C.ARTIFACTS / f"resnet18_cifar10_b{C.BATCH_SIZE}.onnx"
C.export_onnx(model, onnx_path)
onnx.checker.check_model(str(onnx_path))
print("ONNX :", onnx_path, f"({onnx_path.stat().st_size/1e6:.2f} Mo)")
'''),
    md("## 3. Vérification du GPU"),
    code(r'''
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU requis. En local : docker compose run --rm tp ... (avec --gpus all). "
        "Sur Colab : Exécution > Modifier le type d'exécution > GPU."
    )
print("GPU :", torch.cuda.get_device_name(0))
print("mémoire :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "Go")
print("CUDA :", torch.version.cuda, "| cuDNN :", torch.backends.cudnn.version())
'''),
    md("## 4. Contrôle : ONNX Runtime CUDA"),
    code(r'''
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = C.CPU_THREADS
so.inter_op_num_threads = 1
sess_cuda = ort.InferenceSession(
    str(onnx_path),
    sess_options=so,
    providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
)
print("providers :", sess_cuda.get_providers())
fn_cuda = C.make_onnx_fn(sess_cuda, sess_cuda.get_inputs()[0].name)
records, summary, logits_cuda = C.run_benchmark(fn_cuda, x_cpu, "cuda")
C.save_result(
    "03a_gpu_onnx_cuda", records, summary, logits_cuda, labels,
    device="cuda", backend="onnxruntime-cuda",
    notes="Contrôle ONNX Runtime CUDA, même graphe ONNX.",
)
show("03a_gpu_onnx_cuda")
'''),
    md("""
## 5. TensorRT (FP16 + cache de moteur)

L'Execution Provider TensorRT construit un moteur optimisé pour le GPU et la
taille de batch. Le premier appel construit le moteur ; le cache permet de
réutiliser le plan aux exécutions suivantes. Les warmups absorbent la
construction avant les mesures.
"""),
    code(r'''
trt_cache = C.ARTIFACTS / "trt_cache"
trt_cache.mkdir(parents=True, exist_ok=True)
trt_options = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": str(trt_cache),
    "trt_timing_cache_enable": True,
    "trt_timing_cache_path": str(trt_cache),
    "trt_max_workspace_size": 1 * 1024**3,
    "trt_builder_optimization_level": 3,
}
sess_trt = ort.InferenceSession(
    str(onnx_path),
    sess_options=so,
    providers=[
        ("TensorrtExecutionProvider", trt_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
print("providers :", sess_trt.get_providers())
if "TensorrtExecutionProvider" not in sess_trt.get_providers():
    raise RuntimeError(
        "TensorRT EP non chargé. Vérifiez l'installation de tensorrt et la "
        "compatibilité CUDA. Providers disponibles : "
        + str(sess_trt.get_providers())
    )

input_name = sess_trt.get_inputs()[0].name
x_np = x_cpu.numpy()
t0 = time.perf_counter()
sess_trt.run(None, {input_name: x_np[:C.BATCH_SIZE]})
print(f"construction du moteur + premier run : {time.perf_counter() - t0:.2f} s")

fn_trt = C.make_onnx_fn(sess_trt, input_name)
records, summary, logits_trt = C.run_benchmark(fn_trt, x_cpu, "cuda")
C.save_result(
    "03_gpu_onnx_tensorrt", records, summary, logits_trt, labels,
    device="cuda", backend="tensorrt",
    notes="ONNX -> TensorRT EP, FP16, cache de moteur, workspace 1 Go.",
)
show("03_gpu_onnx_tensorrt")
'''),
    md("## 6. Comparaison et intégrité des prédictions"),
    code(r'''
ref = np.load(C.RESULTS / "00_reference_pytorch_gpu_logits.npy")
for rid in ["03a_gpu_onnx_cuda", "03_gpu_onnx_tensorrt"]:
    r = C.load_result(rid)
    logits_r = np.load(C.RESULTS / f"{rid}_logits.npy")
    print(
        f"{r['label']:<34} Top-1={r['accuracy_top1']*100:5.2f} %  "
        f"accord={C.prediction_agreement(logits_r, ref)*100:5.2f} %"
    )
ids = ["00_reference_pytorch_gpu", "03a_gpu_onnx_cuda", "03_gpu_onnx_tensorrt"]
display(pd.DataFrame(C.comparison_rows(ids)).round(4))
bar_metric(ids, "throughput_ips", "throughput_std", "images/s",
           "Pipeline GPU n°1 : débit", color="#4C72B0")
bar_metric(ids, "latency_ms", "latency_std", "ms / image",
           "Pipeline GPU n°1 : latence", color="#C44E52")
'''),
    md("""
### Interprétation

TensorRT doit améliorer le débit GPU par rapport à ONNX Runtime CUDA grâce à la
fusion d'opérateurs et aux kernels spécialisés. Le mode FP16 peut introduire de
très petites différences numériques ; l'accord de prédictions avec la référence
GPU permet de vérifier que la Top-1 n'est pas dégradée. La mémoire GPU pic et
l'énergie par image complètent l'analyse du compromis performance/coût.
"""),
]


NOTEBOOK_04 = [
    md("""
# 04 — Pipeline GPU n°2 : PyTorch → torch.compile / TorchScript → CUDA → GPU

| Couche | Technologie |
|---|---|
| Haut niveau | PyTorch, ResNet-18 + tête 10 classes |
| Format du modèle | Graphe PyTorch 2 / Inductor / TorchScript |
| Middle level | Compilation et fusion d'opérateurs par PyTorch |
| Bas niveau | CUDA / cuDNN et kernels générés par Triton/Inductor |
| Matériel | GPU NVIDIA |

Le notebook mesure TorchScript CUDA (contrôle) puis `torch.compile` (Inductor +
Triton) sur le même modèle et les mêmes entrées.
"""),
    code(SETUP),
    code(HELPERS),
    md("## 1. Données et modèle"),
    code(DATA),
    code(MODEL),
    md("## 2. Vérification du GPU"),
    code(r'''
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU requis. En local : docker compose run --rm tp ... (avec --gpus all). "
        "Sur Colab : Exécution > Modifier le type d'exécution > GPU."
    )
print("GPU :", torch.cuda.get_device_name(0))
print("CUDA :", torch.version.cuda, "| cuDNN :", torch.backends.cudnn.version())
'''),
    md("## 3. TorchScript CUDA (contrôle)"),
    code(r'''
model_gpu = C.build_model("cuda")
example = torch.randn(C.BATCH_SIZE, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, device="cuda")
traced = torch.jit.trace(model_gpu, example)
traced = torch.jit.freeze(traced.eval())
traced = torch.jit.optimize_for_inference(traced)
ts_path = C.ARTIFACTS / "resnet18_cifar10_torchscript_gpu.pt"
torch.jit.save(traced, ts_path)
print("TorchScript GPU :", ts_path, f"({ts_path.stat().st_size/1e6:.2f} Mo)")

fn_ts = C.make_torch_fn(traced, "cuda")
records, summary, logits_ts = C.run_benchmark(fn_ts, x_cpu, "cuda")
C.save_result(
    "04a_gpu_torchscript", records, summary, logits_ts, labels,
    device="cuda", backend="torchscript-cuda",
    notes="TorchScript CUDA, contrôle.",
)
show("04a_gpu_torchscript")
'''),
    md("""
## 4. torch.compile (Inductor + Triton)

`torch.compile` capture le graphe, le compile avec Inductor et génère des
kernels Triton pour le GPU. La première exécution compile ; les warmups
absorbent ce coût avant les mesures.
"""),
    code(r'''
compiled = torch.compile(model_gpu, mode="default", fullgraph=False, dynamic=False)
fn_compiled = C.make_torch_fn(compiled, "cuda")
records, summary, logits_compiled = C.run_benchmark(fn_compiled, x_cpu, "cuda")
C.save_result(
    "04_gpu_torch_compile", records, summary, logits_compiled, labels,
    device="cuda", backend="torch.compile/inductor-cuda",
    notes="torch.compile mode=default, Inductor + Triton, FP32.",
)
show("04_gpu_torch_compile")
'''),
    md("## 5. Comparaison et intégrité des prédictions"),
    code(r'''
ref = np.load(C.RESULTS / "00_reference_pytorch_gpu_logits.npy")
for rid in ["04a_gpu_torchscript", "04_gpu_torch_compile"]:
    r = C.load_result(rid)
    logits_r = np.load(C.RESULTS / f"{rid}_logits.npy")
    print(
        f"{r['label']:<36} Top-1={r['accuracy_top1']*100:5.2f} %  "
        f"accord={C.prediction_agreement(logits_r, ref)*100:5.2f} %"
    )
ids = ["00_reference_pytorch_gpu", "04a_gpu_torchscript", "04_gpu_torch_compile"]
display(pd.DataFrame(C.comparison_rows(ids)).round(4))
bar_metric(ids, "throughput_ips", "throughput_std", "images/s",
           "Pipeline GPU n°2 : débit", color="#55A868")
bar_metric(ids, "latency_ms", "latency_std", "ms / image",
           "Pipeline GPU n°2 : latence", color="#C44E52")
'''),
    md("""
### Interprétation

`torch.compile` peut fusionner des opérateurs et réduire les lancements de
kernels, ce qui améliore souvent le débit sur GPU. Le gain dépend de la taille
de batch, du modèle et du temps de compilation. L'accord de prédictions avec la
référence GPU vérifie que les kernels générés conservent les décisions du
modèle. La comparaison avec TensorRT (notebook 03) montre deux philosophies :
un runtime externe spécialisé vs une compilation intégrée à PyTorch.
"""),
]


def main() -> None:
    write_nb("00_reference_pytorch.ipynb", NOTEBOOK_00)
    write_nb("01_cpu_onnx_openvino.ipynb", NOTEBOOK_01)
    write_nb("02_cpu_torchscript_compile.ipynb", NOTEBOOK_02)
    write_nb("03_gpu_onnx_tensorrt.ipynb", NOTEBOOK_03)
    write_nb("04_gpu_torch_compile.ipynb", NOTEBOOK_04)


if __name__ == "__main__":
    main()
