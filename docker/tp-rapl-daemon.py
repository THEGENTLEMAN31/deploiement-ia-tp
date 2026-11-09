#!/usr/bin/env python3
"""Petit démon RAPL lancé en root par l'entrypoint.

Il expose l'énergie du package CPU sur une socket Unix accessible à l'UID hôte.
Cela permet aux kernels Jupyter (qui ferment les descripteurs hérités) et aux
scripts de mesurer l'énergie CPU sans tourner en root.
"""
from __future__ import annotations

import json
import os
import socket
import sys

BASE = "/sys/class/powercap/intel-rapl:0"


def read_attr(name: str) -> int:
    with open(f"{BASE}/{name}", encoding="ascii") as fh:
        return int(fh.read().strip())


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tp-rapl.sock"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, 0o666)
    server.listen(8)
    while True:
        conn, _ = server.accept()
        with conn:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            try:
                request = json.loads(data or b"{}")
                if request.get("cmd") == "read":
                    response = {
                        "energy_uj": read_attr("energy_uj"),
                        "max_energy_range_uj": read_attr("max_energy_range_uj"),
                    }
                else:
                    response = {"error": "unknown command"}
            except Exception as exc:  # pragma: no cover
                response = {"error": str(exc)}
            conn.sendall((json.dumps(response) + "\n").encode())


if __name__ == "__main__":
    main()
