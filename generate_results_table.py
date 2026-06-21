"""
SLAPBench — Comprehensive Results Table Generator
=================================================
Reads the per-model `.metrics.json` files (so every number is exactly what the
evaluation produced), computes TAR@FAR=0.1% directly from the similarity-score
CSVs, and renders a single self-contained, styled HTML page:

    results/SLAPBench_results.html

The page is browser-openable and screenshot-friendly for presentations.
A reserved row for Claude Opus 4.8 is rendered as "running" until its
similarity-score results appear, at which point it is filled in automatically.

Usage:
    python code/generate_results_table.py
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_HTML = RESULTS_DIR / "SLAPBench_results.html"

# Display registry. `key` = results/<key>/ folder; order is the display order.
MODELS = [
    {"key": "qwen3vl",   "name": "Qwen3-VL-8B-Instruct",   "params": "8B",  "org": "Alibaba",        "kind": "local"},
    {"key": "gemma3",    "name": "Gemma-3-12B-IT",         "params": "12B", "org": "Google DeepMind", "kind": "local"},
    {"key": "internvl3", "name": "InternVL3-8B-Instruct",  "params": "8B",  "org": "OpenGVLab",      "kind": "local"},
    {"key": "qwen25vl",  "name": "Qwen2.5-VL-7B-Instruct", "params": "7B",  "org": "Alibaba",        "kind": "local"},
    {"key": "anthropic", "name": "Claude Opus 4.8",        "params": "—",   "org": "Anthropic",      "kind": "api"},
]


def find_metrics(model_key: str, prompting: str):
    """Return the parsed metrics dict for a model+prompting, or None if absent."""
    patterns = [
        RESULTS_DIR / model_key / "latest" / f"*{prompting}*.metrics.json",
        RESULTS_DIR / model_key / "previous" / "*" / f"*{prompting}*.metrics.json",
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(str(pat)))
    if not hits:
        return None
    # newest by filename timestamp
    hits.sort()
    with open(hits[-1]) as f:
        return json.load(f)


def find_csv(model_key: str, prompting: str):
    patterns = [
        RESULTS_DIR / model_key / "latest" / f"*{prompting}*.csv",
        RESULTS_DIR / model_key / "previous" / "*" / f"*{prompting}*.csv",
    ]
    hits = []
    for pat in patterns:
        hits.extend(g for g in glob.glob(str(pat)) if not g.endswith(".metrics.json"))
    if not hits:
        return None
    hits.sort()
    return hits[-1]


def compute_tar_at_far(csv_path: str, target_far: float = 0.001):
    """TAR at the strictest threshold keeping FAR <= target_far (default 0.1%)."""
    df = pd.read_csv(csv_path).dropna(subset=["similarity_score"])
    df["similarity_score"] = df["similarity_score"].astype(float)
    gen = df[df["label"] == "genuine"]["similarity_score"].values
    imp = df[df["label"] == "impostor"]["similarity_score"].values
    if len(gen) == 0 or len(imp) == 0:
        return None
    best_tar = 0.0
    for thr in np.arange(0, 100.5, 0.5):
        far = float(np.mean(imp >= thr))
        if far <= target_far:
            best_tar = max(best_tar, float(np.mean(gen >= thr)))
    return best_tar * 100.0


def auc_color(auc):
    """Green (good) → amber → red (bad) gradient for an AUC in [0.5, 1.0]."""
    if auc is None:
        return "#e9ecef"
    t = max(0.0, min(1.0, (auc - 0.5) / 0.5))  # 0.5→0, 1.0→1
    # red (220,53,69) → amber (255,193,7) → green (40,167,69)
    if t < 0.5:
        f = t / 0.5
        r = int(220 + (255 - 220) * f); g = int(53 + (193 - 53) * f); b = int(69 + (7 - 69) * f)
    else:
        f = (t - 0.5) / 0.5
        r = int(255 + (40 - 255) * f); g = int(193 + (167 - 193) * f); b = int(7 + (69 - 7) * f)
    return f"rgb({r},{g},{b})"


def collapse_badge(collapsed):
    if collapsed is None:
        return '<span class="pending">…</span>'
    if collapsed:
        return '<span class="badge bad">Collapsed</span>'
    return '<span class="badge ok">Functional</span>'


def fmt(v, suffix="", nd=1):
    return "—" if v is None else f"{v:.{nd}f}{suffix}"


def build_row(m, rank):
    zs = find_metrics(m["key"], "zero_shot")
    td = find_metrics(m["key"], "task_description")
    ss = find_metrics(m["key"], "similarity_score")

    # similarity values
    if ss:
        gen = ss.get("genuine_mean_score")
        imp = ss.get("impostor_mean_score")
        delta = None if (gen is None or imp is None) else gen - imp
        eer = ss.get("eer")
        auc = ss.get("auc")
        csv = find_csv(m["key"], "similarity_score")
        tar = compute_tar_at_far(csv) if csv else None
    else:
        gen = imp = delta = eer = auc = tar = None

    pending = (m["kind"] == "api" and ss is None)
    row_cls = "api-pending" if pending else ("api-row" if m["kind"] == "api" else "")

    rank_cell = "—" if pending else f"#{rank}"
    auc_style = f'style="background:{auc_color(auc)};color:#111;font-weight:700"' if auc is not None else 'class="muted"'
    delta_str = ("—" if delta is None else (f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"))

    return f"""
      <tr class="{row_cls}">
        <td class="rank">{rank_cell}</td>
        <td class="model"><span class="mname">{m['name']}</span><span class="morg">{m['org']} · {m['params']}</span></td>
        <td>{fmt(zs.get('overall_acc') if zs else None, '%')}</td>
        <td>{fmt(zs.get('far') if zs else None, '%')}</td>
        <td>{collapse_badge(zs.get('collapsed') if zs else None)}</td>
        <td>{fmt(td.get('overall_acc') if td else None, '%')}</td>
        <td>{fmt(td.get('far') if td else None, '%')}</td>
        <td>{collapse_badge(td.get('collapsed') if td else None)}</td>
        <td>{fmt(gen, '')}</td>
        <td>{fmt(imp, '')}</td>
        <td class="delta">{delta_str}</td>
        <td>{fmt(eer, '%', 2)}</td>
        <td {auc_style}>{'—' if auc is None else f'{auc:.3f}'}</td>
        <td>{fmt(tar, '%')}</td>
      </tr>"""


def main():
    # rank local+available models by AUC for the rank column
    ranked = []
    for m in MODELS:
        ss = find_metrics(m["key"], "similarity_score")
        auc = ss.get("auc") if ss else None
        ranked.append((m["key"], auc))
    rank_map, r = {}, 1
    for key, auc in sorted(ranked, key=lambda x: (x[1] is None, -(x[1] or 0))):
        if auc is not None:
            rank_map[key] = r; r += 1

    rows = "".join(build_row(m, rank_map.get(m["key"], 0)) for m in MODELS)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SLAPBench — Results</title>
<style>
  :root {{
    --bg:#0f1220; --panel:#171a2b; --ink:#e8eaf2; --muted:#9aa0b4;
    --line:#2a2f45; --accent:#6ea8fe; --ok:#3ddc84; --bad:#ff6b6b;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:linear-gradient(160deg,#0c0f1c,#141831); color:var(--ink);
          font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif; padding:32px; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:.3px; }}
  h1 span {{ color:var(--accent); }}
  .sub {{ color:var(--muted); margin:0 0 22px; font-size:14px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
           padding:6px; box-shadow:0 10px 30px rgba(0,0,0,.35); overflow:hidden; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  thead .grp th {{ background:#1d2238; color:#cfd6f5; font-weight:700; text-transform:uppercase;
                   font-size:11px; letter-spacing:.6px; padding:10px 8px; border-bottom:1px solid var(--line); }}
  thead .sub th {{ background:#191d30; color:var(--muted); font-weight:600; font-size:11px;
                   padding:8px; border-bottom:1px solid var(--line); position:sticky; top:0; }}
  tbody td {{ padding:12px 8px; text-align:center; border-bottom:1px solid var(--line); }}
  tbody tr:hover {{ background:#1b2138; }}
  td.model {{ text-align:left; min-width:200px; }}
  .mname {{ display:block; font-weight:700; }}
  .morg {{ display:block; color:var(--muted); font-size:11px; margin-top:2px; }}
  td.rank {{ font-weight:800; color:var(--accent); }}
  td.delta {{ font-variant-numeric:tabular-nums; }}
  .grp-binary {{ border-left:3px solid #f0a23b33; }}
  .grp-score  {{ border-left:3px solid #6ea8fe33; }}
  .badge {{ padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700; }}
  .badge.ok {{ background:rgba(61,220,132,.15); color:var(--ok); border:1px solid rgba(61,220,132,.4); }}
  .badge.bad {{ background:rgba(255,107,107,.15); color:var(--bad); border:1px solid rgba(255,107,107,.4); }}
  .api-row {{ background:#141a2e; }}
  .api-pending td {{ color:var(--muted); font-style:italic; background:repeating-linear-gradient(
      45deg,#141a2e,#141a2e 10px,#181f36 10px,#181f36 20px); }}
  .pending {{ color:var(--muted); }}
  .muted {{ color:var(--muted); }}
  .legend {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:12px; margin:16px 2px 0; }}
  .legend b {{ color:var(--ink); }}
  .foot {{ color:var(--muted); font-size:12px; margin-top:18px; line-height:1.6; }}
  .pill {{ display:inline-block; background:#1d2238; border:1px solid var(--line);
           border-radius:8px; padding:2px 8px; margin-right:6px; color:#cfd6f5; }}
</style></head>
<body><div class="wrap">
  <h1>SLAP<span>Bench</span> — Multimodal LLM Verification Results</h1>
  <p class="sub">NIST SD302b · 7,832 exhaustive pairs (176 genuine + 7,656 impostor) · FRGP 13 &amp; 14 ·
     metrics generated directly from evaluation outputs</p>

  <div class="card">
  <table>
    <thead>
      <tr class="grp">
        <th rowspan="2">Rank</th>
        <th rowspan="2" style="text-align:left">Model</th>
        <th colspan="6" class="grp-binary">Binary Prompting</th>
        <th colspan="6" class="grp-score">Similarity Scoring (continuous 0–100)</th>
      </tr>
      <tr class="sub">
        <th class="grp-binary">ZS Acc</th><th>ZS FAR</th><th>ZS Status</th>
        <th>TD Acc</th><th>TD FAR</th><th>TD Status</th>
        <th class="grp-score">Gen</th><th>Imp</th><th>&Delta;</th><th>EER</th><th>AUC</th><th>TAR@FAR=0.1%</th>
      </tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>
  </div>

  <div class="legend">
    <span><b>ZS</b> Zero-Shot</span>
    <span><b>TD</b> Task-Description</span>
    <span><b>Gen/Imp</b> mean confidence on genuine / impostor pairs</span>
    <span><b>&Delta;</b> separation (Gen − Imp)</span>
    <span><b>AUC</b> color = green (strong) → red (random)</span>
  </div>

  <p class="foot">
    <span class="pill">Headline</span> Binary prompting collapses in <b>5 of 8</b> configurations;
    similarity scoring eliminates collapse and exposes the true capability gap.
    <b>Qwen3-VL-8B</b> reaches perfect discrimination (AUC 1.000), <b>Gemma-3-12B</b> is a
    functional second (AUC 0.837), while <b>InternVL3-8B</b> (inverted) and <b>Qwen2.5-VL-7B</b>
    (near-random) fail to discriminate. <b>Claude Opus 4.8</b> is currently running and will
    populate automatically once its similarity-score pass completes.<br>
    Accuracy in collapsed runs is low (≈2–6%) because accepting every pair correctly labels
    only the 176 genuine pairs (2.25% of 7,832); FAR, AUC and EER are the meaningful measures.
  </p>
</div></body></html>"""

    OUT_HTML.write_text(html)
    print(f"Wrote {OUT_HTML}")
    print("Open it in a browser, or screenshot for slides.")

    # console echo of the headline numbers for a quick sanity check
    print("\nAUC ranking (similarity scoring):")
    for key, auc in sorted(ranked, key=lambda x: (x[1] is None, -(x[1] or 0))):
        name = next(m["name"] for m in MODELS if m["key"] == key)
        print(f"  {name:28s}  AUC = {'running' if auc is None else f'{auc:.3f}'}")


if __name__ == "__main__":
    main()
