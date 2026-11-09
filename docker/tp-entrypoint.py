#!/usr/bin/env python3
"""Entrypoint du conteneur TP.

Le conteneur démarre en root uniquement pour :
  1. ouvrir les fichiers RAPL (énergie CPU) qui sont autrement lisibles par root ;
  2. préparer les répertoires de cache/HOME ;
puis il abandonne immédiatement les privilèges (setgid/setuid) vers l'UID
de l'utilisateur hôte. Les fichiers créés dans /workspace restent donc
possédés par l'utilisateur, sans ``sudo`` ni ``chown`` manuel.

Les descripteurs RAPL ouverts sont passés au processus enfant via des
numéros de descripteurs stables (>= 100) et la variable d'environnement
RAPL_FDS (JSON).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

RAPL_DIR = Path("/sys/class/powercap/intel-rapl:0")
RAPL_FILES = {
    "energy_uj": RAPL_DIR / "energy_uj",
    "max_energy_range_uj": RAPL_DIR / "max_energy_range_uj",
}


def open_rapl() -> dict[str, int]:
    fds: dict[str, int] = {}
    next_fd = 100
    for key, path in RAPL_FILES.items():
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        os.dup2(fd, next_fd)
        os.close(fd)
        os.set_inheritable(next_fd, True)
        fds[key] = next_fd
        next_fd += 1
    return fds


def main() -> None:
    uid = int(os.environ.get("TP_UID", "1000"))
    gid = int(os.environ.get("TP_GID", "1000"))

    fds = open_rapl()

    # Démon RAPL accessible aux processus non-root (kernels Jupyter inclus).
    socket_path = "/tmp/tp-rapl.sock"
    if Path("/sys/class/powercap/intel-rapl:0/energy_uj").exists():
        subprocess.Popen(
            [sys.executable, "/usr/local/bin/tp-rapl-daemon", socket_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(50):
            if Path(socket_path).exists():
                break
            time.sleep(0.02)
        os.environ["TP_RAPL_SOCKET"] = socket_path

    # Prépare les répertoires que les bibliothèques écrivent au runtime.
    for directory in (
        "/workspace/.home",
        "/workspace/.cache",
        "/workspace/.cache/torch",
        "/workspace/.cache/matplotlib",
        "/workspace/.cache/openvino",
        "/workspace/results",
        "/workspace/report/figures",
        "/workspace/report/screenshots",
    ):
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.chown(p, uid, gid)
        except PermissionError:
            pass

    os.environ["RAPL_FDS"] = json.dumps(fds)
    os.environ.setdefault("HOME", "/workspace/.home")
    os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
    os.environ.setdefault("MPLCONFIGDIR", "/workspace/.cache/matplotlib")

    # Abandon des privilèges avant d'exécuter la commande demandée.
    try:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    except PermissionError:
        # Si le conteneur est déjà lancé en non-root, on continue.
        pass

    cmd = sys.argv[1:] or ["bash"]
    os.execvpe(cmd[0], cmd, os.environ)


if __name__ == "__main__":
    main()
