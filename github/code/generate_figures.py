"""
SLAPBench — Figure Generator (score distributions + ROC curves)
===============================================================
Reproduces the two paper figures for EVERY model that has a completed
similarity-scoring run (all four local models + Claude Opus 4.8 when present).

Raw score arrays come from each model's similarity-scoring CSV; the AUC / EER /
Delta labels are read from the matching `.metrics.json` so the figures agree
exactly with Table~\\ref{tab:main_results} in the paper.

Outputs (PDF + PNG) into manuscript/figures/:
    fig_score_distributions.{pdf,png}
    fig_roc_curves.{pdf,png}

Usage:
    python code/generate_figures.py
"""

import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Display order = by similarity AUC (best first). `disp` is the on-figure label.
MODELS = [
    {"key": "qwen3vl",   "disp": "Qwen3-VL-8B",   "color": "#1f77b4"},
    {"key": "anthropic", "disp": "Claude Opus 4.8", "color": "#9467bd"},
    {"key": "gemma3",    "disp": "Gemma-3-12B",   "color": "#2ca02c"},
    {"key": "internvl3", "disp": "InternVL3-8B",  "color": "#d62728"},
    {"key": "qwen25vl",  "disp": "Qwen2.5-VL-7B", "color": "#ff7f0e"},
]

GEN_COLOR = "#3b6fb0"   # genuine (blue)
IMP_COLOR = "#c0392b"   # impostor (red)


def _latest(patterns):
    hits = []
    for pat in patterns:
        hits.extend(g for g in glob.glob(str(pat)) if not g.endswith(".metrics.json"))
    return sorted(hits)[-1] if hits else None


def load_model(key):
    """Return dict with scores + metrics, or None if the SS run is absent."""
    csv = _latest([
        RESULTS_DIR / key / "latest" / "*similarity_score*.csv",
        RESULTS_DIR / key / "previous" / "*" / "*similarity_score*.csv",
    ])
    mjson = None
    for pat in [RESULTS_DIR / key / "latest" / "*similarity_score*.metrics.json",
                RESULTS_DIR / key / "previous" / "*" / "*similarity_score*.metrics.json"]:
        h = sorted(glob.glob(str(pat)))
        if h:
            mjson = h[-1]
            break
    if not csv or not mjson:
        return None
    df = pd.read_csv(csv).dropna(subset=["similarity_score"])
    df["similarity_score"] = df["similarity_score"].astype(float)
    with open(mjson) as f:
        m = json.load(f)
    return {
        "gen": df[df.label == "genuine"]["similarity_score"].values,
        "imp": df[df.label == "impostor"]["similarity_score"].values,
        "auc": m["auc"], "eer": m["eer"],
        "gmean": m["genuine_mean_score"], "imean": m["impostor_mean_score"],
    }


def roc_points(gen, imp):
    """Sweep thresholds → (FAR, TAR) arrays for an ROC curve."""
    thr = np.linspace(0, 100, 1001)
    far = np.array([np.mean(imp >= t) for t in thr])
    tar = np.array([np.mean(gen >= t) for t in thr])
    # sort by FAR ascending for a clean line
    order = np.argsort(far)
    return far[order], tar[order]


def make_score_distributions(models):
    n = len(models)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows), squeeze=False)
    bins = np.arange(0, 102, 4)
    for i, m in enumerate(models):
        ax = axes[i // cols][i % cols]
        d = m["_data"]
        ax.hist(d["imp"], bins=bins, weights=np.ones_like(d["imp"]) / len(d["imp"]),
                color=IMP_COLOR, alpha=0.75, label=f"Impostor (n={len(d['imp']):,})")
        ax.hist(d["gen"], bins=bins, weights=np.ones_like(d["gen"]) / len(d["gen"]),
                color=GEN_COLOR, alpha=0.75, label=f"Genuine (n={len(d['gen']):,})")
        ax.axvline(d["imean"], color=IMP_COLOR, ls="--", lw=1.5)
        ax.axvline(d["gmean"], color=GEN_COLOR, ls="--", lw=1.5)
        delta = d["gmean"] - d["imean"]
        ax.text(0.03, 0.97,
                f"AUC = {d['auc']:.3f}\nEER = {d['eer']:.1f}%\n$\\Delta$ = {delta:+.1f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_title(m["disp"], fontweight="bold")
        ax.set_xlim(0, 100)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        if i % cols == 0:
            ax.set_ylabel("Fraction of pairs (within class)")
        ax.set_xlabel("Confidence: same person (0–100)")
    # hide any unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_score_distributions.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_roc(models):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    inset = ax.inset_axes([0.34, 0.44, 0.44, 0.44])
    for m in models:
        d = m["_data"]
        far, tar = roc_points(d["gen"], d["imp"])
        lbl = f"{m['disp']}  (AUC = {d['auc']:.3f})"
        ax.plot(far, tar, color=m["color"], lw=2, label=lbl)
        inset.plot(far, tar, color=m["color"], lw=2)
    ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1)
    ax.axvline(0.001, ls=":", color="0.4", lw=1, label="FAR = 0.1%")
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR)")
    ax.set_title("ROC Curves — Similarity-Scoring Prompt", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    # inset: low-FAR operating region
    inset.set_xlim(0, 0.05); inset.set_ylim(0, 1.02)
    inset.set_title("Low-FAR region (FAR $\\leq$ 0.05)", fontsize=8)
    inset.set_xlabel("FAR", fontsize=8); inset.set_ylabel("TAR", fontsize=8)
    inset.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_roc_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    avail = []
    for m in MODELS:
        d = load_model(m["key"])
        if d is None:
            print(f"  skip {m['disp']:18s} — no completed similarity-scoring run")
            continue
        m = dict(m); m["_data"] = d
        avail.append(m)
        print(f"  include {m['disp']:18s} AUC={d['auc']:.3f}  EER={d['eer']:.2f}%")
    if not avail:
        print("No similarity-scoring results found; nothing to plot.")
        return
    make_score_distributions(avail)
    make_roc(avail)
    print(f"\nWrote {len(avail)}-model figures to {FIG_DIR}/")
    print("  fig_score_distributions.{pdf,png}")
    print("  fig_roc_curves.{pdf,png}")


if __name__ == "__main__":
    main()
