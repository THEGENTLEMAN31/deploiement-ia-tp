#!/usr/bin/env python3
"""Rend une page « terminal » avec les preuves d'exécution Docker/GPU.

Les fichiers de log sont produits par scripts/run_all.sh (ou à la main) dans
report/logs/. La page est ensuite capturée avec Chromium/Playwright.
"""
from __future__ import annotations

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "report" / "logs"
OUT = ROOT / "report" / "terminal.html"

SECTIONS = [
    ("Docker et GPU", ["docker-version.txt", "nvidia-smi.txt", "torch-env.txt"]),
    ("Préparation des données", ["prepare.txt"]),
    ("Exécution des notebooks", ["notebooks.txt"]),
]


def main() -> None:
    blocks = []
    for title, files in SECTIONS:
        for filename in files:
            path = LOGS / filename
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            blocks.append(
                f"""
                <div class="term">
                  <div class="bar"><span class="dot red"></span><span class="dot yellow"></span>
                  <span class="dot green"></span><span class="title">{html.escape(title)} — {html.escape(filename)}</span></div>
                  <pre>{html.escape(content)}</pre>
                </div>
                """
            )
    if not blocks:
        print("Aucun log dans report/logs/.")
        return
    page = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Preuves d'exécution — Docker / GPU</title>
<style>
  body {{ background: #eef1f5; margin: 0; font-family: "DejaVu Sans", Arial, sans-serif; }}
  .wrap {{ width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ color: #1f3a5f; font-size: 26px; margin: 0 0 6px 0; }}
  .sub {{ color: #5b6770; margin-bottom: 18px; }}
  .term {{ background: #111820; border-radius: 8px; margin-bottom: 18px;
           box-shadow: 0 6px 18px rgba(0,0,0,.18); overflow: hidden; }}
  .bar {{ background: #1f2933; padding: 8px 12px; display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
  .red {{ background: #ff5f56; }} .yellow {{ background: #ffbd2e; }} .green {{ background: #27c93f; }}
  .title {{ color: #a9b7c6; font-size: 12px; margin-left: 10px; }}
  pre {{ color: #d7e3f0; margin: 0; padding: 14px 16px; font-family: "DejaVu Sans Mono", monospace;
         font-size: 11.5px; line-height: 1.45; white-space: pre-wrap; }}
</style></head><body><div class="wrap">
<h1>Preuves d'exécution — Docker, GPU et notebooks</h1>
<div class="sub">Captures textuelles produites pendant l'exécution réelle du TP.</div>
{''.join(blocks)}
</div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print("écrit :", OUT)


if __name__ == "__main__":
    main()
