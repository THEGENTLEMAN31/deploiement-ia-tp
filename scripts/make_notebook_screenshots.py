#!/usr/bin/env python3
"""Convertit les notebooks exécutés en pages HTML prêtes pour la capture d'écran.

Le rendu imite Jupyter (cellules, prompts, sorties) et ajoute un bandeau de
résultats clés. Les pages sont ensuite capturées avec Chromium/Playwright.
"""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "report" / "notebook_html"
RESULTS = ROOT / "results"

NOTEBOOKS = {
    "00_reference_pytorch.ipynb": {
        "title": "00 — Référence PyTorch eager",
        "results": ["00_reference_pytorch_cpu", "00_reference_pytorch_gpu"],
    },
    "01_cpu_onnx_openvino.ipynb": {
        "title": "01 — ONNX → OpenVINO (CPU)",
        "results": ["01a_cpu_onnxruntime", "01_cpu_onnx_openvino"],
    },
    "02_cpu_torchscript_compile.ipynb": {
        "title": "02 — TorchScript / torch.compile (CPU)",
        "results": ["02a_cpu_torchscript", "02_cpu_torchscript_compile"],
    },
    "03_gpu_onnx_tensorrt.ipynb": {
        "title": "03 — ONNX → TensorRT (GPU)",
        "results": ["03a_gpu_onnx_cuda", "03_gpu_onnx_tensorrt"],
    },
    "04_gpu_torch_compile.ipynb": {
        "title": "04 — torch.compile (GPU)",
        "results": ["04a_gpu_torchscript", "04_gpu_torch_compile"],
    },
}


def md_to_html(text: str) -> str:
    """Rendu markdown minimal (titres, gras, code, listes, tableaux simples)."""
    lines = text.splitlines()
    out: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            content = stripped[level:].strip()
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h{min(level, 4)}>{inline_md(content)}</h{min(level, 4)}>")
        elif stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline_md(stripped[2:])}</li>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            # Les tableaux markdown sont laissés en texte compact.
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<div class='md-table'>{html.escape(stripped)}</div>")
        elif not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{inline_md(stripped)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def inline_md(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def render_output(output: dict) -> str:
    otype = output.get("output_type")
    if otype == "stream":
        return f"<pre class='stream'>{html.escape(''.join(output.get('text', [])))}</pre>"
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        if "image/png" in data:
            raw = data["image/png"]
            if isinstance(raw, list):
                raw = "".join(raw)
            return f"<img class='plot' src='data:image/png;base64,{raw}'>"
        if "text/html" in data:
            raw = data["text/html"]
            if isinstance(raw, list):
                raw = "".join(raw)
            return f"<div class='html-output'>{raw}</div>"
        if "text/plain" in data:
            raw = data["text/plain"]
            if isinstance(raw, list):
                raw = "".join(raw)
            return f"<pre class='stream'>{html.escape(raw)}</pre>"
    if otype == "error":
        return f"<pre class='error'>{html.escape(output.get('ename', 'Erreur'))}: {html.escape(output.get('evalue', ''))}</pre>"
    return ""


def key_results(result_ids: list[str]) -> str:
    cards = []
    for rid in result_ids:
        path = RESULTS / f"{rid}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        s = data["summary"]
        thr = s["throughput_ips"]
        lat = s["latency_ms_per_image"]
        cards.append(
            f"""
            <div class="card">
              <div class="card-title">{html.escape(data['label'])}</div>
              <div class="card-main">{thr['mean']:.2f} <span>img/s</span></div>
              <div class="card-sub">latence {lat['mean']:.3f} ms/img · σ débit {thr['std']:.2f}</div>
              <div class="card-sub">Top-1 <b>{data['accuracy_top1']*100:.1f} %</b></div>
            </div>
            """
        )
    return f"<div class='cards'>{''.join(cards)}</div>" if cards else ""


def build_page(nb_path: Path, title: str, result_ids: list[str]) -> Path:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells_html: list[str] = []
    first_markdown = True
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            if first_markdown:
                first_markdown = False
                # Le titre est déjà dans l'en-tête de la page.
                lines = source.splitlines()
                source = "\n".join(line for line in lines if not line.strip().startswith("# "))
            if source.strip():
                cells_html.append(f"<div class='markdown' id='cell-{cell_idx}'>{md_to_html(source)}</div>")
        elif cell.get("cell_type") == "code":
            exec_count = cell.get("execution_count", "")
            outputs = "".join(render_output(o) for o in cell.get("outputs", []))
            cells_html.append(
                f"""
                <div class="cell" id="cell-{cell_idx}">
                  <div class="prompt">In [{exec_count}]:</div>
                  <pre class="input">{html.escape(source.strip())}</pre>
                </div>
                {f'<div class="output" id="out-{cell_idx}"><div class="prompt out">Out:</div><div class="output-body">{outputs}</div></div>' if outputs else ''}
                """
            )

    page = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; background: #eef1f5; font-family: "DejaVu Sans", Arial, sans-serif;
          color: #1b1f23; }}
  .page {{ width: 1180px; margin: 0 auto; background: white; min-height: 100vh;
           box-shadow: 0 0 18px rgba(0,0,0,.12); }}
  .top {{ background: #1f3a5f; color: white; padding: 22px 28px 18px 28px; }}
  .top .kicker {{ font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase;
                  color: #a9c7e8; }}
  .top h1 {{ margin: 6px 0 4px 0; font-size: 26px; }}
  .top .sub {{ color: #d6e4f2; font-size: 13px; }}
  .cards {{ display: flex; gap: 12px; padding: 16px 28px 8px 28px; flex-wrap: wrap; }}
  .card {{ flex: 1; min-width: 250px; background: #f7f9fb; border: 1px solid #d8dee4;
           border-radius: 6px; padding: 12px 14px; }}
  .card-title {{ font-size: 12px; color: #2e6da4; font-weight: bold; }}
  .card-main {{ font-size: 26px; font-weight: bold; color: #1f3a5f; margin: 4px 0 2px 0; }}
  .card-main span {{ font-size: 13px; font-weight: normal; color: #5b6770; }}
  .card-sub {{ font-size: 11.5px; color: #5b6770; }}
  .notebook {{ padding: 10px 28px 40px 28px; }}
  .cell {{ display: flex; gap: 10px; margin-top: 12px; }}
  .prompt {{ width: 78px; flex: 0 0 78px; font-family: "DejaVu Sans Mono", monospace;
             font-size: 11px; color: #3070b0; text-align: right; padding-top: 6px; }}
  .prompt.out {{ color: #b04a3a; }}
  pre.input {{ flex: 1; margin: 0; background: #f6f8fa; border: 1px solid #d8dee4;
               border-radius: 4px; padding: 10px 12px; font-family: "DejaVu Sans Mono", monospace;
               font-size: 11.5px; line-height: 1.45; white-space: pre-wrap; overflow-x: auto; }}
  .output {{ display: flex; gap: 10px; margin-top: 4px; }}
  .output-body {{ flex: 1; min-width: 0; }}
  pre.stream {{ background: #fbfbfb; border-left: 3px solid #d8dee4; margin: 4px 0;
                padding: 8px 10px; font-family: "DejaVu Sans Mono", monospace;
                font-size: 11px; white-space: pre-wrap; overflow-x: auto; }}
  pre.error {{ background: #fdf0ef; color: #a12b1f; border-left: 3px solid #c44e52; }}
  .html-output {{ font-size: 11px; overflow-x: auto; }}
  .html-output table {{ border-collapse: collapse; margin: 4px 0; }}
  .html-output th, .html-output td {{ border: 1px solid #d8dee4; padding: 4px 7px; }}
  .html-output th {{ background: #eef5fb; }}
  img.plot {{ max-width: 100%; border: 1px solid #d8dee4; border-radius: 4px; margin: 6px 0; }}
  .markdown {{ margin: 14px 0 4px 0; font-size: 13px; }}
  .markdown h1, .markdown h2, .markdown h3 {{ color: #1f3a5f; margin: 12px 0 4px 0; }}
  .markdown p {{ margin: 4px 0; }}
  .markdown code {{ background: #eef2f6; padding: 1px 4px; border-radius: 3px;
                    font-family: "DejaVu Sans Mono", monospace; font-size: 11px; }}
  .md-table {{ font-family: "DejaVu Sans Mono", monospace; font-size: 10.5px;
               color: #5b6770; margin: 2px 0; }}
</style></head>
<body>
<div class="page">
  <div class="top">
    <div class="kicker">TP déploiement IA · notebook exécuté dans Docker</div>
    <h1>{html.escape(title)}</h1>
    <div class="sub">ResNet-18 · CIFAR-10 · 50 images · batch 10 · 5 répétitions · mesures CPU/GPU</div>
  </div>
  {key_results(result_ids)}
  <div class="notebook">
    {''.join(cells_html)}
  </div>
</div>
</body></html>"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (nb_path.stem + ".html")
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    for filename, info in NOTEBOOKS.items():
        nb_path = NB_DIR / filename
        if not nb_path.exists():
            print("absent :", nb_path)
            continue
        out = build_page(nb_path, info["title"], info["results"])
        print("écrit :", out)


if __name__ == "__main__":
    main()
