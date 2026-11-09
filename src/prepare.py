"""Préparation des données et du classifieur commun aux quatre pipelines.

Pourquoi un linear probe ?
--------------------------
ResNet-18 est pré-entraîné sur ImageNet (1000 classes) alors que le TP impose
CIFAR-10 (10 classes). Utiliser directement la sortie 1000 classes ne permet
pas de calculer une Top-1 sur 10 classes. Nous conservons donc le backbone
ImageNet **gelé** et nous entraînons uniquement une tête linéaire 512 -> 10 sur
CIFAR-10. Tous les pipelines partagent ensuite exactement les mêmes poids, ce
qui rend la comparaison des optimisations équitable.

Ce script produit :
  artifacts/cifar10_subset.npz      50 images de test (5 par classe)
  artifacts/classifier_head.npz     poids de la tête linéaire
  artifacts/prepare_metadata.json   protocole et accuracies
  report/figures/cifar10_subset.png aperçu des 50 images
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10

from src import common

SEED = common.SEED
TRAIN_EPOCHS = int(__import__("os").environ.get("TP_HEAD_EPOCHS", "40"))
VAL_SIZE = 5000


def select_subset(targets: np.ndarray, per_class: int = 5) -> np.ndarray:
    """Sélectionne exactement per_class images de test par classe (seed fixe)."""
    rng = np.random.default_rng(SEED)
    indices: list[int] = []
    for cls in range(len(common.CLASSES)):
        cls_indices = np.flatnonzero(targets == cls)
        chosen = rng.choice(cls_indices, size=per_class, replace=False)
        indices.extend(int(i) for i in sorted(chosen))
    return np.asarray(indices, dtype=np.int64)


def extract_features(
    backbone: nn.Module,
    images: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    label: str = "",
) -> np.ndarray:
    """Extrait les features 512-d du backbone gelé, par lots."""
    features: list[np.ndarray] = []
    n = len(images)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            chunk = images[start : start + batch_size]
            x = common.preprocess_images(chunk, device="cpu", chunk_size=128)
            x = x.to(device, non_blocking=True)
            features.append(backbone(x).detach().float().cpu().numpy())
            if start % (batch_size * 20) == 0:
                done = min(start + batch_size, n)
                print(f"  {label}: {done}/{n} images ({done / (time.perf_counter() - t0):.0f} img/s)")
    return np.concatenate(features, axis=0)


def train_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> tuple[nn.Linear, dict]:
    """Entraîne une régression logistique (tête linéaire) sur features gelées."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(features))
    val_idx = perm[:VAL_SIZE]
    train_idx = perm[VAL_SIZE:]

    x_train = torch.from_numpy(features[train_idx]).float().to(device)
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    x_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)

    head = nn.Linear(features.shape[1], len(common.CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_val = -1.0
    history: list[dict] = []
    batch_size = 1024

    for epoch in range(TRAIN_EPOCHS):
        head.train()
        order = torch.randperm(len(x_train), device=device)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(idx)
        scheduler.step()

        head.eval()
        with torch.inference_mode():
            val_pred = head(x_val).argmax(dim=1)
            val_acc = float((val_pred == y_val).float().mean())
        history.append(
            {"epoch": epoch + 1, "train_loss": total_loss / len(x_train), "val_acc": val_acc}
        )
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1:02d}/{TRAIN_EPOCHS} "
                f"loss={history[-1]['train_loss']:.4f} val_acc={val_acc:.4f}"
            )

    head.load_state_dict(best_state)
    head.eval()
    return head, {
        "best_val_accuracy": best_val,
        "epochs": TRAIN_EPOCHS,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "history": history,
    }


def make_preview(images: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    """Génère une planche des 50 images, une ligne par classe."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(10, 5, figsize=(6.5, 12.5))
    for cls, cls_name in enumerate(common.CLASSES):
        cls_indices = np.flatnonzero(labels == cls)
        for j, idx in enumerate(cls_indices[:5]):
            ax = axes[cls, j]
            ax.imshow(images[idx])
            ax.axis("off")
            if j == 0:
                ax.set_title(cls_name, loc="left", fontsize=9, fontweight="bold")
    fig.suptitle("Sous-ensemble CIFAR-10 : 5 images de test par classe", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    common.ensure_dirs()
    common.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Threads CPU: {common.CPU_THREADS}")

    # 1. Données CIFAR-10
    data_root = common.ARTIFACTS / "cifar10"
    train_ds = CIFAR10(root=str(data_root), train=True, download=True)
    test_ds = CIFAR10(root=str(data_root), train=False, download=True)
    train_images = train_ds.data  # uint8 (50000, 32, 32, 3)
    train_labels = np.asarray(train_ds.targets, dtype=np.int64)
    test_images = test_ds.data
    test_labels = np.asarray(test_ds.targets, dtype=np.int64)
    print(f"CIFAR-10: train={len(train_images)}, test={len(test_images)}")

    # 2. Sous-ensemble de 50 images (5 par classe)
    subset_idx = select_subset(test_labels, per_class=5)
    subset_images = test_images[subset_idx]
    subset_labels = test_labels[subset_idx]
    np.savez_compressed(
        common.ARTIFACTS / "cifar10_subset.npz",
        images=subset_images,
        labels=subset_labels,
        test_indices=subset_idx,
    )
    make_preview(subset_images, subset_labels, common.FIGURES / "cifar10_subset.png")
    print(f"Sous-ensemble: {len(subset_images)} images -> artifacts/cifar10_subset.npz")

    # 3. Extraction des features (backbone ImageNet gelé)
    backbone = common.build_backbone(device)
    print("Extraction des features d'entraînement...")
    train_features = extract_features(backbone, train_images, device, label="train")
    print("Extraction des features de test...")
    test_features = extract_features(backbone, test_images, device, label="test")
    print(f"Features: train={train_features.shape}, test={test_features.shape}")

    # 4. Tête linéaire 10 classes
    print("Entraînement du linear probe...")
    head, train_info = train_linear_head(train_features, train_labels, device)

    with torch.inference_mode():
        head_cpu = head.to("cpu")
        test_logits = head_cpu(torch.from_numpy(test_features).float()).numpy()
        subset_logits = test_logits[subset_idx]
    full_acc = common.top1_accuracy(test_logits, test_labels)
    subset_acc = common.top1_accuracy(subset_logits, subset_labels)
    print(f"Accuracy full test (10k): {full_acc:.4f}")
    print(f"Accuracy subset (50):    {subset_acc:.4f}")

    np.savez(
        common.ARTIFACTS / "classifier_head.npz",
        weight=head_cpu.weight.detach().numpy(),
        bias=head_cpu.bias.detach().numpy(),
    )

    metadata = {
        "model": "resnet18",
        "weights": "IMAGENET1K_V1",
        "adaptation": "linear probe (backbone gelé) sur CIFAR-10",
        "feature_dim": int(head_cpu.in_features),
        "num_classes": len(common.CLASSES),
        "classes": common.CLASSES,
        "seed": SEED,
        "subset_size": int(len(subset_images)),
        "subset_test_indices": subset_idx.tolist(),
        "full_test_accuracy": full_acc,
        "subset50_accuracy": subset_acc,
        **train_info,
        "environment": common.environment_info(),
    }
    (common.ARTIFACTS / "prepare_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Artefacts écrits dans artifacts/.")


if __name__ == "__main__":
    main()
