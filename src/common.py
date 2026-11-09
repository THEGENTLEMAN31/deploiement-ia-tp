"""Utilitaires partagés par les quatre pipelines du TP.

Le module centralise :
  - le chargement du sous-ensemble CIFAR-10 et du classifieur 10 classes ;
  - le prétraitement ResNet-18 (32x32 -> 224x224, normalisation ImageNet) ;
  - la construction du modèle PyTorch de référence ;
  - le chronométrage, les statistiques et la mesure d'énergie (RAPL + NVML) ;
  - la sauvegarde des résultats et des prédictions.

Tous les notebooks importent ce module afin de garantir un protocole strictement
identique entre les pipelines.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(os.environ.get("TP_ROOT", Path(__file__).resolve().parents[1])).resolve()
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
SCREENSHOTS = ROOT / "report" / "screenshots"

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
SEED = 42

# Paramètres expérimentaux communs (surchargeables par variables d'environnement).
BATCH_SIZE = int(os.environ.get("TP_BATCH_SIZE", "10"))
REPETITIONS = int(os.environ.get("TP_REPETITIONS", "5"))
WARMUP = int(os.environ.get("TP_WARMUP", "5"))
N_IMAGES = 50
CPU_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))

# Libellés lisibles pour les figures et le rapport.
PIPELINE_LABELS = {
    "00_reference_pytorch_cpu": "PyTorch eager (CPU, réf.)",
    "01a_cpu_onnxruntime": "ONNX Runtime (CPU, contrôle)",
    "01_cpu_onnx_openvino": "ONNX → OpenVINO (CPU)",
    "02a_cpu_torchscript": "TorchScript (CPU, contrôle)",
    "02_cpu_torchscript_compile": "torch.compile (CPU)",
    "00_reference_pytorch_gpu": "PyTorch eager (GPU, réf.)",
    "03a_gpu_onnx_cuda": "ONNX Runtime CUDA (GPU, contrôle)",
    "03_gpu_onnx_tensorrt": "ONNX → TensorRT (GPU)",
    "04a_gpu_torchscript": "TorchScript CUDA (GPU, contrôle)",
    "04_gpu_torch_compile": "torch.compile (GPU)",
}


# --------------------------------------------------------------------------- #
# Configuration / utilitaires généraux
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    for directory in (ARTIFACTS, RESULTS, FIGURES, SCREENSHOTS):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: str | torch.device) -> None:
    """Synchronise CUDA lorsque le device est un GPU."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def is_cuda(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def _cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU inconnu"


def environment_info() -> dict:
    """Instantané des versions logicielles et du matériel, pour le rapport."""
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model_name(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_threads_used": CPU_THREADS,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "batch_size": BATCH_SIZE,
        "repetitions": REPETITIONS,
        "warmup": WARMUP,
        "n_images": N_IMAGES,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import onnx

        info["onnx"] = onnx.__version__
    except Exception:
        pass
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        pass
    try:
        import openvino

        info["openvino"] = openvino.__version__
    except Exception:
        pass
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["gpu"] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(info["gpu"], bytes):
                info["gpu"] = info["gpu"].decode()
            info["gpu_memory_mb"] = round(
                pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2, 1
            )
            info["nvidia_driver"] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(info["nvidia_driver"], bytes):
                info["nvidia_driver"] = info["nvidia_driver"].decode()
        except Exception as exc:  # pragma: no cover - dépend du matériel
            info["gpu_error"] = str(exc)
    return info


# --------------------------------------------------------------------------- #
# Données et modèle
# --------------------------------------------------------------------------- #
def load_subset() -> tuple[np.ndarray, np.ndarray]:
    """Charge les 50 images CIFAR-10 (5 par classe) et leurs labels."""
    path = ARTIFACTS / "cifar10_subset.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent. Lancez d'abord : python -m src.prepare"
        )
    data = np.load(path)
    return data["images"].astype(np.uint8), data["labels"].astype(np.int64)


def preprocess_images(
    images: np.ndarray,
    device: str | torch.device = "cpu",
    chunk_size: int = 256,
) -> torch.Tensor:
    """Prétraitement ResNet-18 : uint8 NCHW 32x32 -> float32 NCHW 224x224 normalisé.

    Le redimensionnement bilinéaire avec antialiasing est appliqué par lots pour
    rester rapide sur les 50 000 images d'entraînement du linear probe.
    """
    device = torch.device(device)
    chunks: list[torch.Tensor] = []
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    for start in range(0, len(images), chunk_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[start : start + chunk_size]))
        batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
        batch = batch.permute(0, 3, 1, 2)
        batch = torch.nn.functional.interpolate(
            batch, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
            align_corners=False, antialias=True,
        )
        batch = (batch - mean) / std
        chunks.append(batch.cpu())
    return torch.cat(chunks, dim=0)


def build_model(device: str | torch.device = "cpu", head_path: Path | None = None) -> nn.Module:
    """ResNet-18 ImageNet + tête linéaire 10 classes (linear probe CIFAR-10)."""
    head_path = head_path or (ARTIFACTS / "classifier_head.npz")
    if not head_path.exists():
        raise FileNotFoundError(
            f"{head_path} absent. Lancez d'abord : python -m src.prepare"
        )
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, len(CLASSES))
    head = np.load(head_path)
    with torch.no_grad():
        model.fc.weight.copy_(torch.from_numpy(head["weight"]))
        model.fc.bias.copy_(torch.from_numpy(head["bias"]))
    model.eval()
    return model.to(device)


def build_backbone(device: str | torch.device = "cpu") -> nn.Module:
    """ResNet-18 ImageNet sans la couche fc, utilisé pour le linear probe."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def export_onnx(
    model: nn.Module,
    path: Path,
    batch_size: int = BATCH_SIZE,
    opset_version: int = 17,
) -> Path:
    """Exporte le modèle en ONNX à batch fixe (batch_size).

    Le batch fixe est volontaire : les 50 images sont traitées par 5 batchs de
    10, ce qui évite tout padding et tout axe dynamique. Les quatre pipelines
    utilisent donc exactement le même graphe et la même taille de batch.
    """
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))
    return path


# --------------------------------------------------------------------------- #
# Fonctions d'inférence (une par runtime)
# --------------------------------------------------------------------------- #
def make_torch_fn(
    model: nn.Module,
    device: str | torch.device,
    batch_size: int = BATCH_SIZE,
) -> Callable[[torch.Tensor], np.ndarray]:
    device = torch.device(device)

    def run(x_cpu: torch.Tensor) -> np.ndarray:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(x_cpu), batch_size):
                xb = x_cpu[start : start + batch_size].to(device)
                outputs.append(model(xb).detach().float().cpu())
        return torch.cat(outputs, dim=0).numpy()

    return run


def make_onnx_fn(session, input_name: str, batch_size: int = BATCH_SIZE):
    def run(x_cpu: torch.Tensor) -> np.ndarray:
        x = x_cpu.numpy()
        outputs = []
        for start in range(0, len(x), batch_size):
            xb = np.ascontiguousarray(x[start : start + batch_size])
            outputs.append(session.run(None, {input_name: xb})[0])
        return np.concatenate(outputs, axis=0)

    return run


def make_openvino_fn(compiled_model, batch_size: int = BATCH_SIZE):
    output_port = compiled_model.output(0)

    def run(x_cpu: torch.Tensor) -> np.ndarray:
        x = x_cpu.numpy()
        outputs = []
        for start in range(0, len(x), batch_size):
            xb = np.ascontiguousarray(x[start : start + batch_size])
            result = compiled_model(xb)
            outputs.append(np.asarray(result[output_port]))
        return np.concatenate(outputs, axis=0)

    return run


# --------------------------------------------------------------------------- #
# Mesure d'énergie
# --------------------------------------------------------------------------- #
def _read_int_fd(fd: int) -> int:
    os.lseek(fd, 0, os.SEEK_SET)
    return int(os.read(fd, 64).decode().strip())


def _read_rapl_socket(path: str) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        sock.connect(path)
        sock.sendall(b'{"cmd": "read"}\n')
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


class EnergyMonitor:
    """Mesure l'énergie CPU (RAPL) et GPU (NVML) autour d'une exécution.

    Le CPU est mesuré via le démon RAPL de l'entrypoint Docker, via des
    descripteurs hérités, ou directement si le processus tourne en root. Le GPU
    utilise le compteur d'énergie total NVML lorsqu'il est disponible, sinon un
    échantillonnage de la puissance instantanée.
    """

    def __init__(self, device: str | torch.device, enabled: bool = True):
        self.device = torch.device(device)
        self.enabled = enabled
        self.cpu_energy_j: float | None = None
        self.gpu_energy_j: float | None = None
        self._cpu_start: int | None = None
        self._cpu_max: int | None = None
        self._gpu_start_mj: int | None = None
        self._gpu_handle = None
        self._pynvml = None
        self._power_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sampled_energy_j = 0.0
        self._prepare()

    def _read_cpu(self) -> tuple[int | None, int | None]:
        """Lit (energy_uj, max_energy_range_uj) via socket, fd hérité ou chemin."""
        if self._rapl_socket:
            try:
                payload = _read_rapl_socket(self._rapl_socket)
                if "energy_uj" in payload:
                    return payload["energy_uj"], payload.get("max_energy_range_uj")
            except Exception:
                pass
        if self._rapl_energy_fd is not None:
            try:
                energy = _read_int_fd(self._rapl_energy_fd)
                maximum = (
                    _read_int_fd(self._rapl_max_fd) if self._rapl_max_fd is not None else None
                )
                return energy, maximum
            except Exception:
                return None, None
        return None, None

    def _prepare(self) -> None:
        if not self.enabled:
            return
        self._rapl_socket = os.environ.get("TP_RAPL_SOCKET") or None
        self._rapl_energy_fd = None
        self._rapl_max_fd = None
        # 1) Démon RAPL (recommandé : fonctionne aussi dans un kernel Jupyter).
        if self._rapl_socket:
            energy, maximum = self._read_cpu()
            if energy is not None:
                self._cpu_start = energy
                self._cpu_max = maximum
                return
        # 2) Descripteurs hérités de l'entrypoint (scripts lancés directement).
        try:
            fds = json.loads(os.environ.get("RAPL_FDS", "{}"))
            if "energy_uj" in fds:
                self._rapl_energy_fd = fds["energy_uj"]
                self._rapl_max_fd = fds.get("max_energy_range_uj")
                energy, maximum = self._read_cpu()
                self._cpu_start = energy
                self._cpu_max = maximum
                return
        except Exception:
            pass
        # 3) Lecture directe si le processus est root.
        try:
            path = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
            max_path = Path("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
            if path.exists():
                self._rapl_energy_fd = os.open(path, os.O_RDONLY)
                self._rapl_max_fd = os.open(max_path, os.O_RDONLY) if max_path.exists() else None
                energy, maximum = self._read_cpu()
                self._cpu_start = energy
                self._cpu_max = maximum
        except Exception:
            self._cpu_start = None

    def _prepare_gpu(self) -> None:
        if not is_cuda(self.device):
            return
        try:
            import pynvml

            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                torch.cuda.current_device()
            )
            try:
                self._gpu_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                    self._gpu_handle
                )
            except Exception:
                self._gpu_start_mj = None
        except Exception:
            self._gpu_handle = None

    def _sample_power(self) -> None:
        last = time.perf_counter()
        while not self._stop_event.is_set():
            time.sleep(0.02)
            now = time.perf_counter()
            try:
                power_w = self._pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
                self._sampled_energy_j += power_w * (now - last)
            except Exception:
                pass
            last = now

    def start(self) -> None:
        if not self.enabled:
            return
        self._prepare_gpu()
        if self._gpu_handle is not None and self._gpu_start_mj is None:
            self._stop_event.clear()
            self._power_thread = threading.Thread(target=self._sample_power, daemon=True)
            self._power_thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        # CPU RAPL
        if self._cpu_start is not None:
            try:
                end, _ = self._read_cpu()
                if end is None:
                    raise RuntimeError("RAPL indisponible")
                delta = end - self._cpu_start
                if delta < 0 and self._cpu_max:
                    delta += self._cpu_max
                self.cpu_energy_j = delta / 1e6
            except Exception:
                self.cpu_energy_j = None
        # GPU NVML
        if self._gpu_handle is not None:
            if self._gpu_start_mj is not None:
                try:
                    end_mj = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(
                        self._gpu_handle
                    )
                    self.gpu_energy_j = (end_mj - self._gpu_start_mj) / 1000.0
                except Exception:
                    self.gpu_energy_j = None
            else:
                self._stop_event.set()
                if self._power_thread is not None:
                    self._power_thread.join(timeout=1.0)
                self.gpu_energy_j = self._sampled_energy_j


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    repetition: int
    n_images: int
    total_s: float
    throughput_ips: float
    latency_ms_per_image: float
    energy_cpu_j: float | None
    energy_gpu_j: float | None
    energy_per_image_cpu_j: float | None
    energy_per_image_gpu_j: float | None
    max_gpu_mem_mb: float | None


def _stats(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "n": len(vals),
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "median": float(statistics.median(vals)),
    }


def run_benchmark(
    fn: Callable[[torch.Tensor], np.ndarray],
    x_cpu: torch.Tensor,
    device: str | torch.device,
    repetitions: int = REPETITIONS,
    warmup: int = WARMUP,
    batch_size: int = BATCH_SIZE,
    measure_energy: bool = True,
) -> tuple[list[RunRecord], dict, np.ndarray]:
    """Exécute warmup + N répétitions indépendantes et renvoie les mesures."""
    n_images = len(x_cpu)
    # Warmup : compile les kernels, construit les moteurs, remplit les caches.
    for _ in range(warmup):
        fn(x_cpu)
        sync(device)

    records: list[RunRecord] = []
    last_output: np.ndarray | None = None
    for rep in range(repetitions):
        if is_cuda(device):
            torch.cuda.reset_peak_memory_stats()
        sync(device)
        monitor = EnergyMonitor(device, enabled=measure_energy)
        monitor.start()
        t0 = time.perf_counter()
        output = fn(x_cpu)
        sync(device)
        t1 = time.perf_counter()
        monitor.stop()

        total_s = t1 - t0
        max_mem = None
        if is_cuda(device):
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
        records.append(
            RunRecord(
                repetition=rep + 1,
                n_images=n_images,
                total_s=total_s,
                throughput_ips=n_images / total_s,
                latency_ms_per_image=total_s * 1000.0 / n_images,
                energy_cpu_j=monitor.cpu_energy_j,
                energy_gpu_j=monitor.gpu_energy_j,
                energy_per_image_cpu_j=(
                    monitor.cpu_energy_j / n_images if monitor.cpu_energy_j else None
                ),
                energy_per_image_gpu_j=(
                    monitor.gpu_energy_j / n_images if monitor.gpu_energy_j else None
                ),
                max_gpu_mem_mb=max_mem,
            )
        )
        last_output = output

    summary = {
        "throughput_ips": _stats([r.throughput_ips for r in records]),
        "latency_ms_per_image": _stats([r.latency_ms_per_image for r in records]),
        "total_s": _stats([r.total_s for r in records]),
        "energy_cpu_j": _stats([r.energy_cpu_j for r in records if r.energy_cpu_j is not None]),
        "energy_gpu_j": _stats([r.energy_gpu_j for r in records if r.energy_gpu_j is not None]),
        "energy_per_image_cpu_j": _stats(
            [r.energy_per_image_cpu_j for r in records if r.energy_per_image_cpu_j is not None]
        ),
        "energy_per_image_gpu_j": _stats(
            [r.energy_per_image_gpu_j for r in records if r.energy_per_image_gpu_j is not None]
        ),
        "max_gpu_mem_mb": _stats(
            [r.max_gpu_mem_mb for r in records if r.max_gpu_mem_mb is not None]
        ),
    }
    assert last_output is not None
    return records, summary, last_output


def top1_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def prediction_agreement(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    return float((logits_a.argmax(axis=1) == logits_b.argmax(axis=1)).mean())


def confusion_matrix(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for truth, pred in zip(labels, logits.argmax(axis=1)):
        cm[int(truth), int(pred)] += 1
    return cm


# --------------------------------------------------------------------------- #
# Sauvegarde
# --------------------------------------------------------------------------- #
def save_result(
    result_id: str,
    records: list[RunRecord],
    summary: dict,
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    backend: str,
    notes: str = "",
    extra: dict | None = None,
) -> Path:
    """Sauvegarde un JSON de résultats + les logits (pour l'accord inter-pipelines)."""
    ensure_dirs()
    payload = {
        "id": result_id,
        "label": PIPELINE_LABELS.get(result_id, result_id),
        "device": device,
        "backend": backend,
        "notes": notes,
        "environment": environment_info(),
        "config": {
            "batch_size": BATCH_SIZE,
            "repetitions": REPETITIONS,
            "warmup": WARMUP,
            "n_images": int(len(labels)),
            "image_size": IMAGE_SIZE,
            "seed": SEED,
        },
        "records": [asdict(r) for r in records],
        "summary": summary,
        "accuracy_top1": top1_accuracy(logits, labels),
        "confusion_matrix": confusion_matrix(logits, labels).tolist(),
    }
    if extra:
        payload.update(extra)
    out_json = RESULTS / f"{result_id}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(RESULTS / f"{result_id}_logits.npy", logits)
    np.save(RESULTS / f"{result_id}_labels.npy", labels)
    return out_json


def load_result(result_id: str) -> dict:
    return json.loads((RESULTS / f"{result_id}.json").read_text(encoding="utf-8"))


def result_exists(result_id: str) -> bool:
    return (RESULTS / f"{result_id}.json").exists()


def load_reference_logits(device: str) -> np.ndarray | None:
    path = RESULTS / f"00_reference_pytorch_{device}_logits.npy"
    return np.load(path) if path.exists() else None


def ensure_reference(
    device: str,
    x_cpu: torch.Tensor,
    labels: np.ndarray,
) -> dict:
    """Charge (ou calcule) la référence PyTorch eager pour un device donné."""
    result_id = f"00_reference_pytorch_{device}"
    if result_exists(result_id):
        return load_result(result_id)
    model = build_model(device)
    fn = make_torch_fn(model, device)
    records, summary, logits = run_benchmark(fn, x_cpu, device)
    save_result(
        result_id,
        records,
        summary,
        logits,
        labels,
        device=device,
        backend="pytorch-eager",
        notes="Référence PyTorch eager, même protocole que les pipelines optimisés.",
    )
    return load_result(result_id)


SUMMARY_METRICS = [
    ("throughput_ips", "Débit (images/s)"),
    ("latency_ms_per_image", "Latence (ms/image)"),
    ("total_s", "Temps total (s)"),
    ("energy_cpu_j", "Énergie CPU (J)"),
    ("energy_gpu_j", "Énergie GPU (J)"),
    ("energy_per_image_cpu_j", "Énergie CPU/image (J)"),
    ("energy_per_image_gpu_j", "Énergie GPU/image (J)"),
    ("max_gpu_mem_mb", "Mémoire GPU pic (Mo)"),
]


def comparison_rows(result_ids: Iterable[str]) -> list[dict]:
    """Lignes prêtes pour pandas/Matplotlib à partir de plusieurs JSON."""
    rows = []
    for result_id in result_ids:
        if not result_exists(result_id):
            continue
        result = load_result(result_id)
        s = result["summary"]
        rows.append(
            {
                "id": result_id,
                "pipeline": result["label"],
                "device": result["device"],
                "backend": result["backend"],
                "throughput_ips": s["throughput_ips"]["mean"],
                "throughput_std": s["throughput_ips"]["std"],
                "latency_ms": s["latency_ms_per_image"]["mean"],
                "latency_std": s["latency_ms_per_image"]["std"],
                "accuracy": result["accuracy_top1"],
                "energy_per_image_j": (
                    s["energy_per_image_gpu_j"]["mean"]
                    if s["energy_per_image_gpu_j"]["n"]
                    else s["energy_per_image_cpu_j"]["mean"]
                    if s["energy_per_image_cpu_j"]["n"]
                    else None
                ),
                "energy_per_image_std": (
                    s["energy_per_image_gpu_j"]["std"]
                    if s["energy_per_image_gpu_j"]["n"]
                    else s["energy_per_image_cpu_j"]["std"]
                    if s["energy_per_image_cpu_j"]["n"]
                    else None
                ),
            }
        )
    return rows


def summary_rows(result: dict) -> list[dict]:
    rows = []
    for key, label in SUMMARY_METRICS:
        st = result.get("summary", {}).get(key, {})
        if st.get("n", 0):
            rows.append(
                {
                    "Métrique": label,
                    "Moyenne": st["mean"],
                    "Écart-type": st["std"],
                    "Min": st["min"],
                    "Max": st["max"],
                }
            )
    rows.append(
        {
            "Métrique": "Top-1 accuracy",
            "Moyenne": result.get("accuracy_top1"),
            "Écart-type": 0.0,
            "Min": result.get("accuracy_top1"),
            "Max": result.get("accuracy_top1"),
        }
    )
    return rows
