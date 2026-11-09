"""Normalise les mesures d'énergie GPU dans les JSON existants.

Les premières exécutions stockaient ``gpu_energy_j`` pour l'ensemble des passes
(20 passes × 50 images). On le ramène à l'énergie des 50 images du protocole et
on ajoute ``gpu_energy_j_50`` par répétition. Idempotent.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def main() -> None:
    for path in sorted((ROOT / "results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        measurement = data.get("energy_measurement")
        if not measurement:
            continue
        reps = measurement.get("repetitions_data", [])
        passes = int(measurement.get("passes", 1) or 1)
        if not reps or passes <= 1:
            continue
        already = all("gpu_energy_j_50" in rep for rep in reps)
        if not already:
            for rep in reps:
                total = rep.get("gpu_energy_j")
                rep["gpu_energy_j_50"] = total / passes if total is not None else None
            for field in ("mean", "std", "min", "max", "median"):
                if measurement["gpu_energy_j"].get(field) is not None:
                    measurement["gpu_energy_j"][field] /= passes
        values = [rep["gpu_energy_j_50"] for rep in reps if rep.get("gpu_energy_j_50") is not None]
        measurement["gpu_energy_j_50"] = stats(values)
        measurement["normalization"] = "gpu_energy_j et gpu_energy_j_50 = énergie pour 50 images"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("normalisé :", path.name)


if __name__ == "__main__":
    main()
