"""Génère les figures comparatives et le tableau récapitulatif du rapport."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import common as C

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

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
CPU_IDS = [i for i in ORDER if i.endswith("_cpu") or "cpu" in i]
GPU_IDS = [i for i in ORDER if i not in CPU_IDS]
REQUIRED = ["01_cpu_onnx_openvino", "02_cpu_torchscript_compile", "03_gpu_onnx_tensorrt", "04_gpu_torch_compile"]


def existing(ids: list[str]) -> list[str]:
    return [i for i in ids if C.result_exists(i)]


def annotate_bars(ax, bars, fmt="{:.1f}", offset=0.02):
    ymax = max((b.get_height() for b in bars), default=1.0)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, ymax * 1.15)


def grouped_bars(ax, df: pd.DataFrame, metric: str, std_metric: str, ylabel: str, title: str, color_offset: int = 0):
    x = np.arange(len(df))
    colors = [PALETTE[(i + color_offset) % len(PALETTE)] for i in range(len(df))]
    bars = ax.bar(x, df[metric], yerr=df[std_metric], capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pipeline"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    annotate_bars(ax, bars)


def plot_device_panels(metric: str, std_metric: str, ylabel: str, title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
    for ax, ids, panel_title, offset in (
        (axes[0], existing(CPU_IDS), "CPU", 0),
        (axes[1], existing(GPU_IDS), "GPU", 1),
    ):
        df = pd.DataFrame(C.comparison_rows(ids))
        if df.empty:
            ax.set_axis_off()
            continue
        grouped_bars(ax, df, metric, std_metric, ylabel, f"{title} — {panel_title}", offset)
    fig.tight_layout()
    fig.savefig(C.FIGURES / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pipeline_diagram() -> None:
    """Schéma des quatre chaînes : haut niveau → format → middle → bas niveau → matériel."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    rows = [
        ("CPU 1", ["PyTorch\nResNet-18", "ONNX\nopset 17", "OpenVINO\nRuntime", "oneDNN\nkernels CPU", "CPU\nIntel"]),
        ("CPU 2", ["PyTorch\nResNet-18", "TorchScript /\ngraphe PyTorch 2", "Inductor\ncompilation", "kernels C++\nOpenMP", "CPU\nIntel"]),
        ("GPU 1", ["PyTorch\nResNet-18", "ONNX\nopset 17", "TensorRT\nFP16", "CUDA / cuDNN", "GPU\nNVIDIA"]),
        ("GPU 2", ["PyTorch\nResNet-18", "TorchScript /\ngraphe PyTorch 2", "Inductor +\nTriton", "kernels CUDA\nTriton", "GPU\nNVIDIA"]),
    ]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B2", "#8C8C8C"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box_w, box_h = 1.72, 0.62
    for row_idx, (row_label, boxes) in enumerate(rows):
        y = 3.75 - row_idx * 1.02
        ax.text(0.05, y + box_h / 2, row_label, ha="left", va="center",
                fontsize=10, fontweight="bold", color="#1f3a5f")
        for col_idx, label in enumerate(boxes):
            x = 1.0 + col_idx * 1.95
            patch = FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.0, edgecolor="white", facecolor=colors[col_idx], alpha=0.92,
            )
            ax.add_patch(patch)
            ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center",
                    fontsize=7.8, color="white", fontweight="bold")
            if col_idx < len(boxes) - 1:
                arrow = FancyArrowPatch(
                    (x + box_w + 0.02, y + box_h / 2),
                    (x + 1.95 - 0.02, y + box_h / 2),
                    arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#5b6770",
                )
                ax.add_patch(arrow)
    legend = ["Haut niveau", "Format du modèle", "Middle level", "Bas niveau", "Matériel"]
    for idx, (label, color) in enumerate(zip(legend, colors)):
        ax.add_patch(plt.Rectangle((1.0 + idx * 1.95, 0.12), 0.25, 0.16, color=color))
        ax.text(1.28 + idx * 1.95, 0.20, label, va="center", fontsize=7.6, color="#333")
    ax.set_title("Les quatre chaînes de déploiement étudiées", fontsize=12, color="#1f3a5f", pad=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_pipelines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    C.ensure_dirs()
    pipeline_diagram()
    ids = existing(ORDER)
    if not ids:
        raise SystemExit("Aucun résultat : exécutez d'abord les notebooks.")
    df = pd.DataFrame(C.comparison_rows(ids))
    df.to_csv(C.RESULTS / "summary.csv", index=False)

    # 1. Débit
    plot_device_panels(
        "throughput_ips", "throughput_std", "images/s",
        "Débit de traitement", "fig_throughput.png",
    )

    # 2. Latence
    plot_device_panels(
        "latency_ms", "latency_std", "ms / image",
        "Temps d'inférence (latence)", "fig_latency.png",
    )

    # 3. Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, panel_ids, panel_title, offset in (
        (axes[0], existing(CPU_IDS), "CPU", 0),
        (axes[1], existing(GPU_IDS), "GPU", 1),
    ):
        sub = df[df["id"].isin(panel_ids)]
        if sub.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(sub))
        colors = [PALETTE[(i + offset) % len(PALETTE)] for i in range(len(sub))]
        bars = ax.bar(x, sub["accuracy"] * 100, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["pipeline"], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Top-1 (%)")
        ax.set_title(f"Top-1 sur les 50 images — {panel_title}", fontsize=10)
        for bar, value in zip(bars, sub["accuracy"] * 100):
            ax.annotate(f"{value:.1f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Énergie par image (CPU et GPU séparément)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, panel_ids, panel_title, offset in (
        (axes[0], existing(CPU_IDS), "CPU (RAPL package)", 0),
        (axes[1], existing(GPU_IDS), "GPU (NVML)", 1),
    ):
        sub = df[df["id"].isin(panel_ids)].dropna(subset=["energy_per_image_j"])
        if sub.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(sub))
        colors = [PALETTE[(i + offset) % len(PALETTE)] for i in range(len(sub))]
        bars = ax.bar(x, sub["energy_per_image_j"], yerr=sub["energy_per_image_std"],
                      capsize=4, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["pipeline"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("J / image")
        ax.set_title(f"Énergie par image — {panel_title}", fontsize=10)
        annotate_bars(ax, bars, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_energy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5. Variabilité : boxplot des débits par répétition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, panel_ids, panel_title in (
        (axes[0], existing(CPU_IDS), "CPU"),
        (axes[1], existing(GPU_IDS), "GPU"),
    ):
        data, labels = [], []
        for rid in panel_ids:
            result = C.load_result(rid)
            data.append([r["throughput_ips"] for r in result["records"]])
            labels.append(result["label"])
        if not data:
            ax.set_axis_off()
            continue
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("images/s")
        ax.set_title(f"Variabilité du débit sur {C.REPETITIONS} répétitions — {panel_title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_variability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 6. Speedup par rapport à la référence du même device
    rows = []
    for rid in existing(REQUIRED):
        result = C.load_result(rid)
        baseline_id = "00_reference_pytorch_cpu" if result["device"] == "cpu" else "00_reference_pytorch_gpu"
        if not C.result_exists(baseline_id):
            continue
        baseline = C.load_result(baseline_id)
        b = baseline["summary"]["throughput_ips"]["mean"]
        s = result["summary"]["throughput_ips"]["mean"]
        rows.append({"pipeline": result["label"], "speedup": s / b, "device": result["device"]})
    if rows:
        speed = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = [PALETTE[0] if d == "cpu" else PALETTE[1] for d in speed["device"]]
        bars = ax.bar(speed["pipeline"], speed["speedup"], color=colors)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="référence = 1")
        ax.set_ylabel("× plus rapide que PyTorch eager")
        ax.set_title("Gain de débit des quatre pipelines par rapport à leur référence", fontsize=10)
        ax.tick_params(axis="x", rotation=18)
        annotate_bars(ax, bars, fmt="{:.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(C.FIGURES / "fig_speedup.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 7. Matrice d'accord des prédictions
    logit_ids = [i for i in ids if (C.RESULTS / f"{i}_logits.npy").exists()]
    if len(logit_ids) > 1:
        labels = [C.load_result(i)["label"] for i in logit_ids]
        matrix = np.zeros((len(logit_ids), len(logit_ids)))
        for i, a in enumerate(logit_ids):
            logits_a = np.load(C.RESULTS / f"{a}_logits.npy")
            for j, b in enumerate(logit_ids):
                logits_b = np.load(C.RESULTS / f"{b}_logits.npy")
                matrix[i, j] = C.prediction_agreement(logits_a, logits_b) * 100
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        sns.heatmap(
            matrix, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100,
            xticklabels=labels, yticklabels=labels, cbar_kws={"label": "accord (%)"}, ax=ax,
        )
        ax.set_title("Accord des prédictions entre pipelines (%)", fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "fig_agreement.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 8. Matrice de confusion de la référence CPU
    if C.result_exists("00_reference_pytorch_cpu"):
        cm = np.asarray(C.load_result("00_reference_pytorch_cpu")["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=C.CLASSES, yticklabels=C.CLASSES, ax=ax,
        )
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vraie classe")
        ax.set_title("Matrice de confusion — référence CPU (50 images)", fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "fig_confusion.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 9. Tableau de synthèse lisible
    display_cols = ["id", "pipeline", "device", "throughput_ips", "throughput_std",
                    "latency_ms", "latency_std", "accuracy", "energy_per_image_j", "energy_per_image_std"]
    table = df[[c for c in display_cols if c in df.columns]].copy()
    table["accuracy_pct"] = table["accuracy"] * 100
    table = table.drop(columns=["accuracy"])
    table.to_csv(C.RESULTS / "summary_table.csv", index=False)
    print(table.round(4).to_string(index=False))
    print("\nFigures écrites dans", C.FIGURES)
    print("Tableaux écrits dans", C.RESULTS)


if __name__ == "__main__":
    main()
