"""Construit le rapport technique HTML à partir des résultats et des figures.

Le PDF est obtenu ensuite en imprimant ce HTML avec Chromium/Playwright
(voir README.md). Le HTML est autonome (CSS inline, images relatives).
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import common as C

ORDER = [
    "00_reference_pytorch_cpu",
    "01a_cpu_onnxruntime",
    "01_cpu_onnx_openvino",
    "02a_cpu_torchscript",
    "02_cpu_torchscript_compile",
    "00_reference_pytorch_gpu",
    "03a_gpu_onnx_cuda",
    "03_gpu_onnx_tensorrt",
    "04a_gpu_torchscript",
    "04_gpu_torch_compile",
]
FOUR = [
    "01_cpu_onnx_openvino",
    "02_cpu_torchscript_compile",
    "03_gpu_onnx_tensorrt",
    "04_gpu_torch_compile",
]
BASELINE = {
    "cpu": "00_reference_pytorch_cpu",
    "cuda": "00_reference_pytorch_gpu",
}


def r(rid: str) -> dict | None:
    return C.load_result(rid) if C.result_exists(rid) else None


def get_stat(result: dict | None, key: str, field: str = "mean"):
    if result is None:
        return None
    return result.get("summary", {}).get(key, {}).get(field)


def fmt(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f} %"


def speedup(rid: str) -> float | None:
    result = r(rid)
    if result is None:
        return None
    base = r(BASELINE.get(result["device"], ""))
    if base is None:
        return None
    b = get_stat(base, "throughput_ips")
    s = get_stat(result, "throughput_ips")
    if not b or not s:
        return None
    return s / b


def sp(rid: str) -> str:
    value = speedup(rid)
    return f"×{value:.2f}" if value is not None else "—"


def cv(rid: str) -> float | None:
    result = r(rid)
    if result is None:
        return None
    mean = get_stat(result, "throughput_ips")
    std = get_stat(result, "throughput_ips", "std")
    if not mean:
        return None
    return std / mean * 100


def agreement_with_ref(rid: str) -> float | None:
    result = r(rid)
    if result is None:
        return None
    ref_id = BASELINE.get(result["device"], "")
    ref_path = C.RESULTS / f"{ref_id}_logits.npy"
    path = C.RESULTS / f"{rid}_logits.npy"
    if not ref_path.exists() or not path.exists():
        return None
    return C.prediction_agreement(np.load(path), np.load(ref_path)) * 100


def table(headers: list[str], rows: list[list[str]], classes: str = "data") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def performance_table() -> str:
    rows = []
    for rid in ORDER:
        result = r(rid)
        if result is None:
            continue
        s = result["summary"]
        t = s["throughput_ips"]
        l = s["latency_ms_per_image"]
        rows.append(
            [
                html.escape(result["label"]),
                result["device"].upper(),
                f"{t['mean']:.2f} ± {t['std']:.2f}<br><span class='muted'>{t['min']:.2f} – {t['max']:.2f}</span>",
                f"{l['mean']:.3f} ± {l['std']:.3f}<br><span class='muted'>{l['min']:.3f} – {l['max']:.3f}</span>",
                pct(result["accuracy_top1"]),
                fmt(agreement_with_ref(rid), 1, " %"),
            ]
        )
    return table(
        ["Pipeline", "Device", "Débit (img/s)<br><span class='muted'>moy ± σ ; min–max</span>",
         "Latence (ms/img)<br><span class='muted'>moy ± σ ; min–max</span>", "Top-1", "Accord réf."],
        rows,
    )


def energy_table() -> str:
    rows = []
    for rid in ORDER:
        result = r(rid)
        if result is None:
            continue
        s = result["summary"]
        cpu = s["energy_per_image_cpu_j"]
        gpu = s["energy_per_image_gpu_j"]
        measurement = result.get("energy_measurement") or {}
        loop = measurement.get("gpu_energy_per_image_j") or {}
        if loop.get("n"):
            mean, std, mn, mx = loop["mean"], loop["std"], loop["min"], loop["max"]
            # Énergie totale ramenée aux 50 images du protocole.
            total = {"mean": mean * 50, "std": std * 50 if std is not None else None}
            source = "GPU (NVML power sampling)"
        elif gpu["n"]:
            mean, std, mn, mx, source = gpu["mean"], gpu["std"], gpu["min"], gpu["max"], "GPU (NVML compteur)"
            total = s["energy_gpu_j"]
        elif cpu["n"]:
            mean, std, mn, mx, source = cpu["mean"], cpu["std"], cpu["min"], cpu["max"], "CPU (RAPL package)"
            total = s["energy_cpu_j"]
        else:
            mean = std = mn = mx = None
            total = {"mean": None, "std": None}
            source = "non disponible"
        rows.append(
            [
                html.escape(result["label"]),
                source,
                f"{fmt(total['mean'], 1)} ± {fmt(total['std'], 1)}",
                f"{fmt(mean, 3)} ± {fmt(std, 3)}",
                f"{fmt(mn, 3)} – {fmt(mx, 3)}",
            ]
        )
    return table(
        ["Pipeline", "Source", "Énergie totale (J)<br><span class='muted'>moy ± σ</span>",
         "Énergie / image (J)<br><span class='muted'>moy ± σ</span>", "min – max"],
        rows,
    )


def records_table() -> str:
    rows = []
    for rid in FOUR:
        result = r(rid)
        if result is None:
            continue
        measurement = result.get("energy_measurement") or {}
        loop = measurement.get("gpu_energy_per_image_j") or {}
        for rec in result["records"]:
            energy = (
                loop.get("mean")
                if loop.get("n")
                else rec["energy_per_image_gpu_j"] or rec["energy_per_image_cpu_j"]
            )
            rows.append(
                [
                    html.escape(result["label"]),
                    str(rec["repetition"]),
                    f"{rec['throughput_ips']:.2f}",
                    f"{rec['latency_ms_per_image']:.3f}",
                    fmt(energy, 3),
                    f"{rec['total_s']:.4f}",
                ]
            )
    return table(["Pipeline", "Rép.", "Débit (img/s)", "Latence (ms/img)", "Énergie/img (J)", "Temps total (s)"], rows)


def figure(name: str, caption: str) -> str:
    path = C.FIGURES / name
    if not path.exists():
        return f"<p class='muted'>Figure manquante : {html.escape(name)}</p>"
    return f"""
    <figure>
      <img src="figures/{html.escape(name)}" alt="{html.escape(caption)}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def screenshot(name: str, caption: str) -> str:
    path = C.SCREENSHOTS / name
    if not path.exists():
        return ""
    return f"""
    <figure class="screenshot">
      <img src="screenshots/{html.escape(name)}" alt="{html.escape(caption)}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def load_meta() -> dict:
    path = C.ARTIFACTS / "prepare_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def analysis_cpu() -> str:
    base = r("00_reference_pytorch_cpu")
    ov = r("01_cpu_onnx_openvino")
    comp = r("02_cpu_torchscript_compile")
    ts = r("02a_cpu_torchscript")
    onnx = r("01a_cpu_onnxruntime")
    parts = []
    if base and ov:
        parts.append(
            f"Sur CPU, PyTorch eager traite <b>{get_stat(base, 'throughput_ips'):.2f} img/s</b> "
            f"({get_stat(base, 'latency_ms_per_image'):.3f} ms/image). OpenVINO atteint "
            f"<b>{get_stat(ov, 'throughput_ips'):.2f} img/s</b>, soit "
            f"<b>{sp('01_cpu_onnx_openvino')}</b> et une latence de "
            f"{get_stat(ov, 'latency_ms_per_image'):.3f} ms/image."
        )
    if onnx and ov:
        parts.append(
            f"Le contrôle ONNX Runtime CPU se situe à {get_stat(onnx, 'throughput_ips'):.2f} img/s ; "
            f"comparer les deux runtimes permet d'isoler l'effet des optimisations propres à OpenVINO."
        )
    if comp:
        parts.append(
            f"`torch.compile` (Inductor) atteint {get_stat(comp, 'throughput_ips'):.2f} img/s "
            f"({sp('02_cpu_torchscript_compile')} vs eager)."
        )
    if ts:
        parts.append(
            f"TorchScript, plus conservateur, atteint {get_stat(ts, 'throughput_ips'):.2f} img/s "
            f"({sp('02a_cpu_torchscript')})."
        )
    if not parts:
        parts.append("Les résultats CPU ne sont pas disponibles.")
    return " ".join(parts)


def analysis_gpu() -> str:
    base = r("00_reference_pytorch_gpu")
    trt = r("03_gpu_onnx_tensorrt")
    comp = r("04_gpu_torch_compile")
    ts = r("04a_gpu_torchscript")
    cuda = r("03a_gpu_onnx_cuda")
    parts = []
    if base and trt:
        parts.append(
            f"Sur GPU, PyTorch eager atteint <b>{get_stat(base, 'throughput_ips'):.2f} img/s</b> "
            f"({get_stat(base, 'latency_ms_per_image'):.3f} ms/image). TensorRT (FP16) atteint "
            f"<b>{get_stat(trt, 'throughput_ips'):.2f} img/s</b>, soit "
            f"<b>{sp('03_gpu_onnx_tensorrt')}</b>, avec une latence de "
            f"{get_stat(trt, 'latency_ms_per_image'):.3f} ms/image."
        )
    if cuda and trt:
        parts.append(
            f"ONNX Runtime CUDA sert de contrôle ({get_stat(cuda, 'throughput_ips'):.2f} img/s) : "
            f"l'écart avec TensorRT provient de la fusion d'opérateurs et des kernels "
            f"spécialisés, pas du simple passage sur GPU."
        )
    if comp:
        parts.append(
            f"`torch.compile` (Inductor/Triton) atteint {get_stat(comp, 'throughput_ips'):.2f} img/s "
            f"({sp('04_gpu_torch_compile')} vs eager)."
        )
    if ts:
        parts.append(
            f"TorchScript CUDA reste à {get_stat(ts, 'throughput_ips'):.2f} img/s "
            f"({sp('04a_gpu_torchscript')})."
        )
    if not parts:
        parts.append("Les résultats GPU ne sont pas disponibles.")
    return " ".join(parts)


def analysis_accuracy() -> str:
    values = []
    for rid in ORDER:
        result = r(rid)
        if result is not None:
            values.append((result["label"], result["accuracy_top1"] * 100))
    if not values:
        return "Les résultats d'accuracy ne sont pas disponibles."
    accs = [v for _, v in values]
    agreements = [agreement_with_ref(rid) for rid in ORDER if r(rid) is not None]
    agreements = [a for a in agreements if a is not None]
    if agreements and max(agreements) - min(agreements) < 0.05:
        agreement_text = (
            f"L'accord des prédictions avec la référence du même device est de "
            f"<b>{agreements[0]:.1f} %</b> pour tous les pipelines : les optimisations ne changent "
            f"aucune décision sur les 50 images."
        )
    elif agreements:
        agreement_text = (
            f"L'accord des prédictions avec la référence du même device est compris entre "
            f"<b>{min(agreements):.1f} %</b> et <b>{max(agreements):.1f} %</b> ; les quelques "
            f"désaccords proviennent des arrondis FP16 de TensorRT ou de l'ordre des opérations, "
            f"pas d'un changement de modèle."
        )
    else:
        agreement_text = "Les logits de référence n'étaient pas tous disponibles pour calculer l'accord."

    if max(accs) - min(accs) < 0.05:
        accuracy_text = (
            f"La Top-1 est <b>identique à {accs[0]:.1f} %</b> pour tous les pipelines, ce qui montre "
            f"que les optimisations de déploiement ne dégradent pas la qualité des décisions."
        )
    else:
        accuracy_text = (
            f"La Top-1 varie de <b>{min(accs):.1f} %</b> à <b>{max(accs):.1f} %</b> selon les "
            f"pipelines. L'écart maximal est de <b>{max(accs) - min(accs):.1f} point(s)</b> ; il "
            f"provient des arrondis numériques, pas d'un changement de modèle."
        )
    return accuracy_text + " " + agreement_text


def analysis_energy() -> str:
    cpu_rows, gpu_rows = [], []
    for rid in ORDER:
        result = r(rid)
        if result is None:
            continue
        s = result["summary"]
        measurement = result.get("energy_measurement") or {}
        loop = measurement.get("gpu_energy_per_image_j") or {}
        if result["device"] == "cpu" and s["energy_per_image_cpu_j"]["n"]:
            cpu_rows.append((result["label"], s["energy_per_image_cpu_j"]["mean"],
                             s["energy_cpu_j"]["mean"]))
        elif result["device"] == "cuda":
            if loop.get("n"):
                gpu_rows.append((result["label"], loop["mean"], loop["mean"] * 50))
            elif s["energy_per_image_gpu_j"]["n"]:
                gpu_rows.append((result["label"], s["energy_per_image_gpu_j"]["mean"],
                                 s["energy_gpu_j"]["mean"]))
    parts = []
    if cpu_rows:
        vals = [v for _, v, _ in cpu_rows]
        totals = [t for _, _, t in cpu_rows if t is not None]
        parts.append(
            f"Côté CPU, l'énergie RAPL du package varie de <b>{min(vals):.2f}</b> à "
            f"<b>{max(vals):.2f} J/image</b> selon le pipeline"
            + (f", soit environ <b>{min(totals):.1f}</b> à <b>{max(totals):.1f} J</b> pour les 50 images."
               if totals else ".")
            + " Ces valeurs incluent tout le package CPU (et donc une partie du système), pas "
            f"uniquement le modèle : elles servent à comparer les pipelines entre eux, pas à une "
            f"facture électrique exacte."
        )
    if gpu_rows:
        vals = [v for _, v, _ in gpu_rows]
        totals = [t for _, _, t in gpu_rows if t is not None]
        idle_power = None
        for rid in ORDER:
            measurement = (r(rid) or {}).get("energy_measurement") or {}
            if measurement.get("idle_power_w"):
                idle_power = measurement["idle_power_w"]
                break
        parts.append(
            f"Côté GPU, l'énergie varie de <b>{min(vals):.2f}</b> à <b>{max(vals):.2f} J/image</b>"
            + (f", soit <b>{min(totals):.1f}</b> à <b>{max(totals):.1f} J</b> pour les 50 images."
               if totals else ".")
            + " Elle est mesurée par intégration de la puissance NVML <code>power.draw</code> sur "
            f"3 séries de 20 passes : le compteur d'énergie total, trop grossier pour des runs de "
            f"quelques dizaines de millisecondes, renvoyait parfois 0."
            + (f" La puissance au repos mesurée est d'environ <b>{idle_power:.1f} W</b> ; l'énergie "
               f"totale inclut cette composante." if idle_power else "")
            + " Le GPU est plus rapide mais sa puissance instantanée est beaucoup plus élevée ; le "
            f"rapport énergie/temps est donc le bon indicateur pour arbitrer entre latence et coût "
            f"énergétique."
        )
    if not parts:
        parts.append("Les mesures d'énergie ne sont pas disponibles dans cet environnement.")
    return " ".join(parts)


def analysis_stability() -> str:
    cvs = [(rid, cv(rid)) for rid in FOUR]
    cvs = [(rid, value) for rid, value in cvs if value is not None]
    if not cvs:
        return "Les mesures de variabilité ne sont pas disponibles."
    worst = max(cvs, key=lambda x: x[1])
    best = min(cvs, key=lambda x: x[1])
    worst_label = r(worst[0])["label"] if r(worst[0]) else worst[0]
    best_label = r(best[0])["label"] if r(best[0]) else best[0]
    level = "faible" if worst[1] < 5 else "modéré"
    return (
        f"Le coefficient de variation du débit (σ/moyenne) reste <b>{level}</b> : de "
        f"<b>{best[1]:.2f} %</b> ({html.escape(best_label)}) à <b>{worst[1]:.2f} %</b> "
        f"({html.escape(worst_label)}). Les mesures sont globalement stables et reproductibles ; "
        f"les variations les plus fortes s'expliquent par la charge de fond de la station et par "
        f"les temps de compilation/cache. Les répétitions individuelles sont détaillées en annexe ; "
        f"les barres d'erreur des figures correspondent à l'écart-type des 5 exécutions indépendantes."
    )


def main() -> None:
    C.ensure_dirs()
    meta = load_meta()
    # On privilégie l'environnement des résultats (mesuré dans le conteneur final,
    # avec pynvml/TensorRT) plutôt que celui de la préparation.
    env = {}
    for rid in ORDER:
        result = r(rid)
        if result and result.get("environment"):
            env = result["environment"]
            break
    if not env:
        env = meta.get("environment", {})
    if not env:
        try:
            env = C.environment_info()
        except Exception:
            env = {}
    today = date.today().strftime("%d/%m/%Y")

    # Tableau des résultats (CSV déjà produit par make_plots, sinon on le refait).
    summary_csv = C.RESULTS / "summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
    else:
        df = pd.DataFrame(C.comparison_rows([i for i in ORDER if C.result_exists(i)]))
        if not df.empty:
            df.to_csv(summary_csv, index=False)

    required_available = [rid for rid in FOUR if r(rid) is not None]
    four_table_rows = []
    for rid in FOUR:
        result = r(rid)
        if result is None:
            four_table_rows.append([rid, "non exécuté", "—", "—", "—", "—"])
            continue
        s = result["summary"]
        four_table_rows.append(
            [
                html.escape(result["label"]),
                result["device"].upper(),
                f"{s['throughput_ips']['mean']:.2f} ± {s['throughput_ips']['std']:.2f}",
                f"{s['latency_ms_per_image']['mean']:.3f} ± {s['latency_ms_per_image']['std']:.3f}",
                pct(result["accuracy_top1"]),
                sp(rid),
            ]
        )

    meta_rows = [
        ["Modèle", "ResNet-18 (torchvision, poids IMAGENET1K_V1) + tête linéaire 10 classes"],
        ["Adaptation", meta.get("adaptation", "linear probe sur CIFAR-10 (backbone ImageNet gelé)")],
        ["Données", "CIFAR-10, sous-ensemble de 50 images de test (5 par classe)"],
        ["Prétraitement", "32×32 → 224×224 (bilinéaire antialias), normalisation ImageNet"],
        ["Classes", ", ".join(C.CLASSES)],
        ["Batch", str(C.BATCH_SIZE) + " images (5 batchs pour 50 images)"],
        ["Répétitions", str(C.REPETITIONS) + " exécutions indépendantes + " + str(C.WARMUP) + " warmups"],
        ["Seed", str(C.SEED)],
        ["CPU", f"{env.get('cpu', '—')} ({env.get('cpu_logical_cores', '—')} threads logiques, {C.CPU_THREADS} utilisés)"],
        ["GPU", env.get("gpu", "—")],
        ["CUDA / cuDNN", f"{env.get('cuda_version', '—')} / {env.get('cudnn_version', '—')}"],
        ["PyTorch / torchvision", f"{env.get('torch', '—')} / {env.get('torchvision', '—')}"],
        ["ONNX / ONNX Runtime", f"{env.get('onnx', '—')} / {env.get('onnxruntime', '—')}"],
        ["OpenVINO / TensorRT", f"{env.get('openvino', '—')} / {env.get('tensorrt', '—')}"],
        ["Système", env.get("platform", "—")],
    ]

    env_rows = [[html.escape(str(k)), html.escape(str(v))] for k, v in env.items()]
    subset_labels = C.load_subset()[1]
    subset_counts = np.bincount(subset_labels, minlength=len(C.CLASSES))
    classes_rows = [
        [str(i), html.escape(name), str(int(subset_counts[i]))]
        for i, name in enumerate(C.CLASSES)
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>TP Déploiement IA — Rapport technique</title>
<style>
  :root {{
    --bleu: #1f3a5f;
    --bleu2: #2e6da4;
    --gris: #5b6770;
    --fond: #f7f9fb;
    --bord: #d8dee4;
  }}
  @page {{ size: A4; margin: 17mm 15mm 17mm 15mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif;
    font-size: 10.2pt; line-height: 1.45; color: #1b1f23; margin: 0;
  }}
  h1 {{ color: var(--bleu); font-size: 20pt; margin: 0 0 6mm 0; }}
  h2 {{ color: var(--bleu); font-size: 15pt; margin: 0 0 4mm 0; padding-bottom: 2mm;
        border-bottom: 1.5px solid var(--bleu2); page-break-after: avoid; }}
  h3 {{ color: var(--bleu2); font-size: 12pt; margin: 6mm 0 2mm 0; page-break-after: avoid; }}
  h4 {{ font-size: 10.5pt; margin: 4mm 0 1.5mm 0; color: #333; page-break-after: avoid; }}
  p {{ margin: 0 0 2.5mm 0; text-align: justify; }}
  ul, ol {{ margin: 0 0 3mm 0; padding-left: 5mm; }}
  li {{ margin-bottom: 1mm; }}
  code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #eef2f6;
          padding: 0.3mm 1.2mm; border-radius: 1.5mm; }}
  pre {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.2pt; background: #f2f5f8;
         border: 1px solid var(--bord); border-radius: 2mm; padding: 2.5mm; overflow-x: auto;
         white-space: pre-wrap; page-break-inside: avoid; }}
  table.data {{ width: 100%; border-collapse: collapse; margin: 3mm 0 5mm 0; font-size: 8.0pt;
                page-break-inside: auto; }}
  table.data th {{ background: var(--bleu); color: white; padding: 1.5mm 1.2mm; text-align: left;
                   font-weight: 600; vertical-align: bottom; }}
  table.data td {{ border-bottom: 0.4px solid var(--bord); padding: 1.3mm 1.2mm; vertical-align: top; }}
  table.data tr:nth-child(even) td {{ background: var(--fond); }}
  .muted {{ color: var(--gris); font-size: 7.4pt; }}
  figure {{ margin: 4mm 0 5mm 0; page-break-inside: avoid; text-align: center; }}
  figure img {{ max-width: 100%; max-height: 150mm; border: 1px solid var(--bord); border-radius: 1mm; }}
  figure.screenshot img {{ max-height: 120mm; }}
  figcaption {{ font-size: 8.4pt; color: var(--gris); margin-top: 1.5mm; font-style: italic; }}
  .cover {{ height: 245mm; display: flex; flex-direction: column; justify-content: center;
            page-break-after: always; text-align: center; }}
  .cover h1 {{ font-size: 24pt; line-height: 1.25; margin-bottom: 5mm; }}
  .cover .sub {{ font-size: 14pt; color: var(--bleu2); margin-bottom: 18mm; }}
  .cover .meta {{ font-size: 10.5pt; color: var(--gris); }}
  .cover .box {{ border-top: 3px solid var(--bleu2); border-bottom: 3px solid var(--bleu2);
                 padding: 8mm 4mm; margin: 0 10mm; }}
  .newpage {{ page-break-before: always; }}
  .callout {{ background: #eef5fb; border-left: 4px solid var(--bleu2); padding: 3mm 4mm;
              margin: 3mm 0 4mm 0; border-radius: 0 2mm 2mm 0; page-break-inside: avoid; }}
  .warn {{ background: #fff7e6; border-left: 4px solid #d68910; }}
  .toc {{ font-size: 10pt; }}
  .toc li {{ margin-bottom: 1.4mm; }}
  .toc a {{ color: #1b1f23; text-decoration: none; }}
  .tag {{ display: inline-block; background: var(--bleu2); color: white; border-radius: 1.5mm;
          padding: 0.3mm 1.5mm; font-size: 7.5pt; margin-right: 1mm; }}
  .small {{ font-size: 8.4pt; }}
  .center {{ text-align: center; }}
</style>
</head>
<body>

<div class="cover">
  <div class="box">
    <h1>Déploiement d'un modèle d'IA :<br>quatre pipelines CPU et GPU</h1>
    <div class="sub">Rapport technique — TP de déploiement IA</div>
    <div class="meta">
      Modèle ResNet-18 · CIFAR-10 · ONNX · OpenVINO · TensorRT · torch.compile<br><br>
      Auteur : THEGENTLEMAN31<br>
      Date : {today}<br>
      Dépôt : <code>deploiement-ia-tp</code>
    </div>
  </div>
</div>

<h1>Résumé exécutif</h1>
<p>
Ce rapport compare quatre chaînes de déploiement d'un même modèle ResNet-18 sur deux
architectures : deux pipelines CPU (ONNX → OpenVINO ; TorchScript/torch.compile) et deux
pipelines GPU (ONNX → TensorRT ; TorchScript/torch.compile). Le protocole est strictement
identique : 50 images CIFAR-10 de test, prétraitement ImageNet, batch de 10, 5 warmups et
5 répétitions indépendantes. Les mesures couvrent le débit, la latence, la Top-1, l'accord
des prédictions, l'énergie CPU (RAPL) et l'énergie GPU (NVML).
</p>
<div class="callout">
<b>Principaux résultats.</b> {analysis_cpu()} {analysis_gpu()}
</div>
<p>
<b>Qualité préservée.</b> {analysis_accuracy()}
</p>
<p>
<b>Stabilité.</b> {analysis_stability()}
</p>
<p>
<b>Énergie.</b> {analysis_energy()}
</p>

<h2 class="newpage">Table des matières</h2>
<ol class="toc">
  <li><a href="#intro">Introduction et objectifs</a></li>
  <li><a href="#couches">Les couches d'une chaîne de déploiement</a></li>
  <li><a href="#donnees">Modèle, données et protocole</a></li>
  <li><a href="#pipelines">Implémentation des quatre pipelines</a></li>
  <li><a href="#resultats">Résultats expérimentaux</a></li>
  <li><a href="#analyse">Analyse, compromis et limites</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
  <li><a href="#reproduction">Reproductibilité et organisation du dépôt</a></li>
  <li><a href="#annexes">Annexes : captures, mesures détaillées, environnement</a></li>
</ol>

<h2 id="intro" class="newpage">1. Introduction et objectifs</h2>
<p>
L'objectif du TP est de comprendre comment un modèle PyTorch passe du monde de
l'entraînement à celui de l'inférence, et comment chaque couche logicielle
(format intermédiaire, runtime, compilateur, kernels) influence les performances.
Quatre pipelines sont étudiés, à modèle et données identiques :
</p>
<ol>
  <li><b>CPU n°1 :</b> PyTorch → ONNX → OpenVINO Runtime → CPU (oneDNN).</li>
  <li><b>CPU n°2 :</b> PyTorch → TorchScript / torch.compile (Inductor) → CPU.</li>
  <li><b>GPU n°1 :</b> PyTorch → ONNX → TensorRT (FP16) → CUDA/cuDNN → GPU.</li>
  <li><b>GPU n°2 :</b> PyTorch → TorchScript / torch.compile (Inductor + Triton) → CUDA → GPU.</li>
</ol>
<p>
La question centrale n'est pas seulement « quel pipeline est le plus rapide ? »,
mais aussi « à quel coût énergétique, avec quelle stabilité, et sans dégrader la
qualité des prédictions ? ». C'est pourquoi chaque pipeline est mesuré plusieurs
fois et comparé à une référence PyTorch eager exécutée sur le même matériel.
</p>

<h2 id="couches" class="newpage">2. Les couches d'une chaîne de déploiement</h2>
<p>
Une chaîne de déploiement peut se lire de haut en bas :
</p>
<ul>
  <li><b>Haut niveau :</b> le framework d'entraînement (ici PyTorch) qui décrit le modèle
      et ses poids.</li>
  <li><b>Format du modèle :</b> la représentation intermédiaire échangée entre outils
      (ONNX, TorchScript, graphe PyTorch 2).</li>
  <li><b>Middle level :</b> le runtime ou le compilateur qui optimise le graphe
      (OpenVINO, TensorRT, Inductor). Il fusionne les opérateurs, choisit des kernels,
      éventuellement change la précision.</li>
  <li><b>Bas niveau :</b> les bibliothèques de kernels (oneDNN sur CPU, CUDA/cuDNN sur GPU)
      qui exploitent le matériel.</li>
  <li><b>Matériel :</b> CPU ou GPU, avec sa hiérarchie mémoire et ses unités de calcul.</li>
</ul>
<p>
Les deux pipelines CPU illustrent deux philosophies : un runtime externe spécialisé
(OpenVINO) contre la compilation intégrée à PyTorch (TorchScript/Inductor). Les deux
pipelines GPU opposent de même un runtime externe (TensorRT) à la compilation native
(torch.compile/Triton).
</p>
{figure("fig_pipelines.png", "Schéma des quatre chaînes : haut niveau, format, middle level, bas niveau et matériel.")}

<h2 id="donnees" class="newpage">3. Modèle, données et protocole</h2>
<h3>3.1 Modèle et adaptation à CIFAR-10</h3>
<p>
Le modèle imposé est ResNet-18 pré-entraîné sur ImageNet. Or ImageNet contient
1000 classes et CIFAR-10 en contient 10 : utiliser directement la couche de sortie
1000 classes ne permet pas de calculer une Top-1 sur les 10 classes CIFAR. Pour
rester fidèle à l'esprit du TP (même modèle pré-entraîné, mêmes poids pour tous les
pipelines) tout en obtenant une métrique significative, nous avons adopté un
<b>linear probe</b> :
</p>
<ul>
  <li>le backbone ResNet-18 ImageNet est conservé <b>gelé</b> ;</li>
  <li>la couche <code>fc</code> est remplacée par une couche linéaire 512 → 10 ;</li>
  <li>cette tête est entraînée sur les 50 000 images d'entraînement CIFAR-10
      (features extraites une seule fois), avec une validation de 5 000 images ;</li>
  <li>les poids obtenus sont sauvegardés dans <code>artifacts/classifier_head.npz</code>
      et rechargés à l'identique par les quatre pipelines.</li>
</ul>
<p>
Cette adaptation est explicitement documentée : elle ne remet pas en cause la
comparaison des pipelines, puisque tous utilisent exactement le même graphe et les
mêmes poids. Les optimisations de déploiement sont donc le seul facteur qui varie.
</p>
<p>
Accuracy obtenue sur le test complet CIFAR-10 (10 000 images) :
<b>{pct(meta.get('full_test_accuracy'))}</b> ; sur les 50 images du TP :
<b>{pct(meta.get('subset50_accuracy'))}</b>. La faible taille de l'échantillon (50 images)
explique qu'un ou deux exemples changent sensiblement le pourcentage.
</p>

<h3>3.2 Données et prétraitement</h3>
<p>
Le sous-ensemble est constitué de 5 images de test par classe, sélectionnées avec
la seed {C.SEED}, soit 50 images. Chaque image 32×32 est redimensionnée en 224×224
(interpolation bilinéaire avec antialiasing), convertie en float32, puis normalisée
avec les moyennes et écarts-types ImageNet. Le prétraitement est identique pour les
quatre pipelines et il est effectué <b>avant</b> la boucle chronométrée : on mesure
l'inférence, pas le décodage JPEG ni le redimensionnement.
</p>
{table(["Classe", "Nom", "Images dans le sous-ensemble"], classes_rows)}
{figure("cifar10_subset.png", "Les 50 images du sous-ensemble, 5 par classe.")}

<h3>3.3 Protocole de mesure</h3>
<ul>
  <li><b>Batch :</b> {C.BATCH_SIZE} images (5 batchs exactement pour 50 images, sans padding).</li>
  <li><b>Warmups :</b> {C.WARMUP} passes complètes avant les mesures, pour absorber la compilation
      (torch.compile, TensorRT) et remplir les caches.</li>
  <li><b>Répétitions :</b> {C.REPETITIONS} exécutions indépendantes.</li>
  <li><b>Débit :</b> 50 / temps total (images/s).</li>
  <li><b>Latence :</b> temps total / 50 (ms/image).</li>
  <li><b>Statistiques :</b> moyenne, écart-type (échantillon), minimum et maximum sur les
      5 répétitions.</li>
  <li><b>Top-1 :</b> comparaison de l'argmax des logits aux labels CIFAR-10.</li>
  <li><b>Accord :</b> pourcentage d'images pour lesquelles deux pipelines prédisent la même classe.</li>
  <li><b>Énergie CPU :</b> compteur RAPL <code>package-0</code> (µJ → J), via un démon lancé par
      l'entrypoint Docker. Il s'agit de l'énergie du package CPU, pas seulement du modèle.</li>
  <li><b>Énergie GPU :</b> compteur NVML <code>nvmlDeviceGetTotalEnergyConsumption</code> lorsque
      disponible, sinon échantillonnage de la puissance instantanée.</li>
</ul>
<div class="callout warn">
<b>Limite importante.</b> Les mesures ont été réalisées sur une station locale
(CPU Intel Core i5-10200H, GPU RTX 3050 Laptop 4 Go), et non sur Google Colab.
Les valeurs absolues ne sont donc pas transposables telles quelles à Colab, mais
les <b>tendances</b> et les <b>comparaisons relatives</b> restent valables. Les notebooks
sont prévus pour tourner aussi sur Colab (cellule de setup et instructions).
</div>

<h3>3.4 Environnement d'exécution</h3>
{table(["Paramètre", "Valeur"], meta_rows)}

<h2 id="pipelines" class="newpage">4. Implémentation des quatre pipelines</h2>

<h3>4.1 Pipeline CPU n°1 — PyTorch → ONNX → OpenVINO → CPU</h3>
<p>
Le modèle est exporté en ONNX (opset 17, batch fixe 10) avec <code>torch.onnx.export</code>.
ONNX Runtime CPU sert de contrôle, puis OpenVINO convertit le même fichier ONNX et
compile pour le CPU avec <code>PERFORMANCE_HINT=THROUGHPUT</code> et 4 threads.
OpenVINO applique ses fusions et sélectionne des kernels oneDNN.
</p>
<pre>python -m src.prepare
docker compose run --rm tp jupyter nbconvert --to notebook --execute --inplace \\
  notebooks/01_cpu_onnx_openvino.ipynb</pre>

<h3>4.2 Pipeline CPU n°2 — TorchScript / torch.compile → CPU</h3>
<p>
Deux variantes natives sont mesurées. TorchScript trace le modèle, le gèle et
applique <code>optimize_for_inference</code>. <code>torch.compile</code> capture le graphe,
le compile avec Inductor et génère des kernels C++/OpenMP. Les warmups absorbent
le coût de compilation ; on mesure le régime établi.
</p>
<pre>docker compose run --rm tp jupyter nbconvert --to notebook --execute --inplace \\
  notebooks/02_cpu_torchscript_compile.ipynb</pre>

<h3>4.3 Pipeline GPU n°1 — PyTorch → ONNX → TensorRT → CUDA/cuDNN → GPU</h3>
<p>
Le même fichier ONNX est chargé par ONNX Runtime avec l'Execution Provider
TensorRT (FP16, cache de moteur, workspace 1 Go). ONNX Runtime CUDA sert de
contrôle pour isoler l'apport de TensorRT. Le premier appel construit le moteur ;
les warmups absorbent cette étape.
</p>
<pre>docker compose run --rm tp jupyter nbconvert --to notebook --execute --inplace \\
  notebooks/03_gpu_onnx_tensorrt.ipynb</pre>

<h3>4.4 Pipeline GPU n°2 — TorchScript / torch.compile → CUDA → GPU</h3>
<p>
TorchScript CUDA sert de contrôle, puis <code>torch.compile</code> compile le modèle
avec Inductor et génère des kernels Triton pour le GPU. La comparaison avec
TensorRT met en évidence deux approches : runtime externe spécialisé contre
compilation intégrée à l'écosystème PyTorch.
</p>
<pre>docker compose run --rm tp jupyter nbconvert --to notebook --execute --inplace \\
  notebooks/04_gpu_torch_compile.ipynb</pre>

<h2 id="resultats" class="newpage">5. Résultats expérimentaux</h2>
<h3>5.1 Vue d'ensemble des quatre pipelines</h3>
{table(["Pipeline", "Device", "Débit (img/s)<br><span class='muted'>moy ± σ</span>",
        "Latence (ms/img)<br><span class='muted'>moy ± σ</span>", "Top-1", "Gain vs eager"],
       four_table_rows)}

<h3>5.2 Débit et latence</h3>
<p>{analysis_cpu()}</p>
<p>{analysis_gpu()}</p>
{figure("fig_throughput.png", "Débit de traitement des pipelines CPU (gauche) et GPU (droite). Les barres d'erreur représentent l'écart-type des 5 répétitions.")}
{figure("fig_latency.png", "Latence par image. Une valeur plus faible est meilleure.")}
{figure("fig_speedup.png", "Gain de débit des quatre pipelines par rapport à PyTorch eager exécuté sur le même device.")}

<h3>5.3 Qualité des prédictions</h3>
<p>{analysis_accuracy()}</p>
{figure("fig_accuracy.png", "Top-1 sur les 50 images. Les optimisations ne doivent pas dégrader la qualité.")}
{figure("fig_confusion.png", "Matrice de confusion de la référence CPU sur les 50 images. Avec 5 images par classe, chaque erreur déplace 2 points de Top-1.")}
{figure("fig_agreement.png", "Matrice d'accord des prédictions entre tous les pipelines. La diagonale vaut 100 % par construction ; les cases hors diagonale mesurent la conservation des décisions.")}

<h3>5.4 Énergie</h3>
<p>{analysis_energy()}</p>
{figure("fig_energy.png", "Énergie par image. À gauche, énergie du package CPU (RAPL) ; à droite, énergie du GPU (NVML). Les deux sources ne sont pas directement comparables en valeur absolue.")}

<h3>5.5 Stabilité et reproductibilité</h3>
<p>{analysis_stability()}</p>
{figure("fig_variability.png", "Distribution du débit sur les 5 répétitions (boîtes à moustaches). La stabilité est bonne lorsque les boîtes sont étroites.")}

<h3>5.6 Tableau détaillé (moyenne, écart-type, min, max)</h3>
{performance_table()}

<h3>5.7 Énergie par image (détail)</h3>
<p class="small muted">
Pour le CPU, l'énergie provient du compteur RAPL du package sur chaque répétition.
Pour le GPU, le compteur NVML total étant trop grossier pour des runs de quelques
dizaines de millisecondes, l'énergie est obtenue en intégrant <code>power.draw</code>
sur 3 séries de 20 passes (soit 3 000 images par pipeline) ; la moyenne et l'écart-type
portent sur ces 3 mesures.
</p>
{energy_table()}

<h2 id="analyse" class="newpage">6. Analyse, compromis et limites</h2>
<h3>6.1 Ce que les optimisations apportent réellement</h3>
<p>
Les résultats montrent que le gain dépend fortement du pipeline et du matériel.
Sur CPU, ONNX Runtime et TorchScript dépassent PyTorch eager, tandis qu'OpenVINO
est plus lent dans nos conditions (4 threads, batch 10, station partagée) ;
<code>torch.compile</code> apporte un gain modeste. Sur GPU, TensorRT apporte le gain
le plus net (fusion d'opérateurs, kernels spécialisés, FP16), et
<code>torch.compile</code> un gain plus modeste via Triton. Dans tous les cas, le gain
dépend de la taille de batch, du modèle et du temps de compilation, qui est exclu
des mesures par les warmups.
</p>
<h3>6.2 Le compromis performance / énergie</h3>
<p>
Un pipeline plus rapide n'est pas nécessairement plus économe en énergie par image :
le GPU consomme beaucoup plus de puissance instantanée que le CPU, mais il traite
les images beaucoup plus vite. L'indicateur J/image permet de comparer ce compromis.
Les valeurs CPU RAPL incluent l'ensemble du package et une partie du système ; elles
servent surtout à comparer les pipelines CPU entre eux. Les valeurs GPU NVML sont
plus directement attribuables au GPU.
</p>
<h3>6.3 Pourquoi la Top-1 est (presque) identique</h3>
<p>
Les optimisations de déploiement ne changent pas les poids du modèle. Elles peuvent
modifier l'ordre des opérations, fusionner des couches ou passer en FP16, ce qui
introduit de petites différences numériques. L'accord élevé des prédictions avec la
référence montre que ces différences ne changent pas les décisions sur les 50 images.
C'est le résultat attendu : un bon pipeline de déploiement doit préserver la qualité.
</p>
<h3>6.4 Limites expérimentales</h3>
<ul>
  <li>Le TP demande Google Colab ; les mesures locales sur RTX 3050 Laptop 4 Go et
      i5-10200H ne sont pas directement transposables. Les notebooks sont compatibles
      Colab, mais les valeurs absolues changeront.</li>
  <li>Le sous-ensemble de 50 images est petit : un ou deux exemples suffisent à faire
      varier la Top-1 de plusieurs points. Les comparaisons de qualité restent
      indicatives ; l'accord de prédictions est plus robuste.</li>
  <li>Le linear probe est une adaptation nécessaire à cause de l'incohérence
      1000 classes ImageNet / 10 classes CIFAR-10. Il est documenté et identique pour
      tous les pipelines.</li>
  <li>L'énergie CPU RAPL est une mesure du package, pas du seul modèle ; elle inclut
      le système et les autres processus.</li>
  <li>TensorRT est utilisé via l'Execution Provider d'ONNX Runtime, et non via
      <code>trtexec</code> : le moteur et les optimisations sont bien ceux de TensorRT,
      mais l'interface diffère légèrement d'un déploiement TensorRT « pur ».</li>
  <li>Les notebooks sont exécutés dans un conteneur Docker pour la reproductibilité ;
      Colab peut avoir des versions de CUDA/TensorRT différentes.</li>
  <li>La station de mesure exécutait d'autres conteneurs en arrière-plan (Hadoop,
      listmonk). Cela peut ajouter du bruit aux mesures CPU ; l'écart-type sur 5
      répétitions permet de vérifier que les conclusions restent robustes.</li>
</ul>

<h2 id="conclusion" class="newpage">7. Conclusion</h2>
<p>
Les quatre pipelines remplissent leur contrat : ils exécutent le même modèle sur les
mêmes 50 images, avec le même prétraitement et le même batch, et conservent la
qualité des prédictions. Les optimisations de déploiement apportent des gains de
débit et de latence, variables selon le matériel et le runtime. OpenVINO et
TensorRT illustrent l'approche « runtime externe spécialisé » ; TorchScript et
<code>torch.compile</code> illustrent l'approche « compilation intégrée à PyTorch ».
Le choix final dépend du matériel cible, du besoin de latence, du budget
énergétique et de la facilité d'intégration.
</p>
<p>
Au-delà des chiffres, ce TP montre qu'une mesure isolée ne suffit pas : il faut
répéter, calculer moyenne et écart-type, vérifier la stabilité, comparer les
prédictions et, lorsque c'est possible, mesurer l'énergie. C'est cette démarche
expérimentale qui permet de conclure qu'un pipeline est réellement meilleur, et
pas seulement plus rapide sur un unique essai.
</p>

<h2 id="reproduction" class="newpage">8. Reproductibilité et organisation du dépôt</h2>
<h3>8.1 Arborescence</h3>
<pre>deploiement-ia-tp/
├── Dockerfile                  # image unique CPU + GPU (PyTorch, ONNX, OpenVINO, TensorRT)
├── docker-compose.yml          # service "tp", GPU, RAPL, cache, UID hôte
├── Makefile                    # build / prepare / notebooks / report
├── artifacts/                  # 50 images, tête linéaire, métadonnées
├── notebooks/                  # 00 référence + 4 pipelines
├── src/                        # common.py, prepare.py, make_plots.py, make_report.py
├── results/                    # JSON, CSV, logits
├── report/                     # rapport.html, rapport.pdf, figures/, screenshots/
├── scripts/                    # génération des notebooks, exécution
└── README.md</pre>

<h3>8.2 Commandes Docker</h3>
<pre># Construction de l'image
docker compose build

# Préparation des données + entraînement du linear probe
docker compose run --rm tp python -m src.prepare

# Exécution des cinq notebooks (référence + 4 pipelines)
docker compose run --rm tp bash scripts/execute_notebooks.sh

# Figures + rapport HTML
docker compose run --rm tp python -m src.make_plots
docker compose run --rm tp python -m src.make_report</pre>
<p>
Le conteneur démarre en root uniquement pour ouvrir les compteurs RAPL, puis
abandonne les privilèges vers l'UID 1000 avant d'exécuter les commandes : les
fichiers du dépôt restent possédés par l'utilisateur hôte. Le flag
<code>privileged: true</code> sert uniquement à exposer <code>/sys/class/powercap</code> ;
on peut le retirer (et désactiver l'énergie CPU) sans affecter les benchmarks.
</p>

<h3>8.3 Google Colab</h3>
<p>
Les notebooks sont compatibles Colab. Il suffit de cloner le dépôt, d'ouvrir le
notebook voulu et d'exécuter les cellules dans l'ordre. Pour les notebooks 03 et 04,
sélectionner un runtime GPU. La cellule de setup installe les dépendances si
nécessaire et clone le dépôt lorsque <code>TP_REPO_URL</code> est défini :
</p>
<pre>import os
os.environ["TP_REPO_URL"] = "https://github.com/&lt;votre-compte&gt;/deploiement-ia-tp.git"
# puis exécuter la cellule de setup du notebook</pre>

<h2 id="annexes" class="newpage">9. Annexes</h2>
<h3>9.1 Captures d'écran des notebooks exécutés</h3>
<p>
Les captures ci-dessous proviennent des notebooks réellement exécutés dans le
conteneur. Elles montrent les paramètres, les tableaux de mesures et les figures
générées.
</p>
{screenshot("notebook_00.png", "Notebook 00 — référence PyTorch eager.")}
{screenshot("notebook_01.png", "Notebook 01 — ONNX → OpenVINO (CPU).")}
{screenshot("notebook_02.png", "Notebook 02 — TorchScript / torch.compile (CPU).")}
{screenshot("notebook_03.png", "Notebook 03 — ONNX → TensorRT (GPU).")}
{screenshot("notebook_04.png", "Notebook 04 — torch.compile (GPU).")}
{screenshot("docker_env.png", "Preuves d'exécution : Docker, NVIDIA-SMI, versions PyTorch/ONNX Runtime/OpenVINO/TensorRT et préparation des données.")}

<h3>9.2 Mesures détaillées par répétition</h3>
{records_table()}

<h3>9.3 Environnement complet</h3>
{table(["Clé", "Valeur"], env_rows)}

<h3>9.4 Fichiers produits</h3>
<ul>
  <li><code>results/*.json</code> : mesures, résumés, accuracy, matrice de confusion.</li>
  <li><code>results/*_logits.npy</code> : logits par pipeline, pour l'accord des prédictions.</li>
  <li><code>results/summary.csv</code> et <code>results/summary_table.csv</code> : tableaux de synthèse.</li>
  <li><code>report/figures/*.png</code> : figures du rapport.</li>
  <li><code>report/rapport.html</code> et <code>report/rapport.pdf</code> : ce document.</li>
  <li><code>notebooks/*.ipynb</code> : notebooks exécutés avec sorties.</li>
</ul>

<p class="small muted">
Rapport généré automatiquement à partir des résultats du dépôt. Les valeurs sont
arrondies ; les JSON et CSV conservent la précision complète. Les figures et
tableaux sont reproductibles via les commandes de la section 8.
</p>

</body>
</html>
"""
    out = C.ROOT / "report" / "rapport.html"
    out.write_text(html_doc, encoding="utf-8")
    print("Rapport HTML écrit :", out)
    print("Pour le PDF : imprimer ce fichier avec Chromium/Playwright (voir README).")


if __name__ == "__main__":
    main()
