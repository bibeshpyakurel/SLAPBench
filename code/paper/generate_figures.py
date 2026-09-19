"""
Generate figures for the SLAPBench paper (7,832-pair exhaustive run).

  1. Score distribution panels (genuine vs. impostor), one per model.
     - fraction-within-class normalization (handles the 176 vs 7,656 imbalance)
     - discrete-aware bars (model scores are clustered integers, not continuous)
     - per-panel AUC / EER annotation and genuine/impostor mean markers
     - NO fitted Gaussian curves: the score distributions are discrete and
       sometimes single-spike (std = 0) or bimodal, so a normal fit would
       misrepresent the data.
  2. ROC curves for all three models (true staircase) with a low-FAR inset.
  3. Console report of AUC, EER, and TAR@FAR=0.1% for cross-checking the
     metrics JSON / manuscript tables.
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results", "sd302b")
OUT = os.path.join(BASE, "paper", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

GEN_COLOR = "#1565C0"   # blue  — genuine
IMP_COLOR = "#C62828"   # red   — impostor


def latest_ss_csv(model_dir):
    """Return the most recent similarity_score CSV for a model.

    Searches both latest/ and previous/ because the pipeline keeps only the
    most recent run per prompting strategy in latest/; for some models the
    similarity-scoring run was superseded (in latest/) by a later binary run
    and pushed into previous/."""
    patterns = [
        os.path.join(RESULTS, model_dir, "latest", "task8_*_similarity_score_*.csv"),
        os.path.join(RESULTS, model_dir, "previous", "*", "task8_*_similarity_score_*.csv"),
    ]
    matches = sorted(p for pat in patterns for p in glob.glob(pat))
    if not matches:
        raise FileNotFoundError(patterns)
    return matches[-1]


# ordered by similarity-scoring AUC (matches Table I in the paper)
MODELS = [
    {"name": "Qwen3-VL-8B",    "csv": latest_ss_csv("qwen3vl")},
    {"name": "Claude Opus 4.8", "csv": latest_ss_csv("anthropic")},
    {"name": "Gemma-3-12B",    "csv": latest_ss_csv("gemma3")},
    {"name": "InternVL3-8B",   "csv": latest_ss_csv("internvl3")},
    {"name": "Qwen2.5-VL-7B",  "csv": latest_ss_csv("qwen25vl")},
]


# ── Data + metrics ────────────────────────────────────────────────────────────
def load(csv_path):
    df = pd.read_csv(csv_path)
    genuine = df[df["label"] == "genuine"]["similarity_score"].values.astype(float)
    impostor = df[df["label"] == "impostor"]["similarity_score"].values.astype(float)
    labels = df["label"].map({"genuine": 1, "impostor": 0}).values
    scores = df["similarity_score"].values.astype(float)
    return genuine, impostor, labels, scores


def compute_eer(fpr, tpr):
    """EER = point where FAR (=fpr) equals FRR (=1-tpr)."""
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0


def tar_at_far(genuine, impostor, target_far=0.001):
    """Highest TAR achievable while holding FAR <= target_far."""
    thresholds = np.arange(0, 100.6, 0.5)
    best_tar, best_thr = 0.0, None
    for thr in thresholds:
        far = np.mean(impostor >= thr)
        tar = np.mean(genuine >= thr)
        if far <= target_far and tar >= best_tar:
            best_tar, best_thr = tar, thr
    return best_tar, best_thr


# Scores are multiples of 5; one bar per 5-wide bin centered on the value.
bins = np.arange(-2.5, 103, 5)

# Split into two categorized figures so panels are large enough to read:
#   - "discriminating" models (functional separation) -> full-width 1x3 figure
#   - "failed" models (inverted / compressed)         -> single-column 2x1 figure
DISCRIMINATING = ["Qwen3-VL-8B", "Claude Opus 4.8", "Gemma-3-12B"]
FAILED = ["InternVL3-8B", "Qwen2.5-VL-7B"]

# ── pre-pass: compute stats for every model (also feeds the ROC figure) ───────
stats = {}
roc_data = []
print("=" * 64)
print(f"{'Model':16s} {'GenMean':>8s} {'ImpMean':>8s} {'Delta':>7s} "
      f"{'AUC':>6s} {'EER%':>7s} {'TAR@.1%':>8s}")
print("=" * 64)
for m in MODELS:
    genuine, impostor, labels, scores = load(m["csv"])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    eer = compute_eer(fpr, tpr)
    tar, _ = tar_at_far(genuine, impostor)
    g_mean, i_mean = genuine.mean(), impostor.mean()
    stats[m["name"]] = dict(genuine=genuine, impostor=impostor, auc=roc_auc,
                            eer=eer, g_mean=g_mean, i_mean=i_mean)
    roc_data.append((m["name"], fpr, tpr, roc_auc))
    print(f"{m['name']:16s} {g_mean:8.1f} {i_mean:8.1f} {g_mean-i_mean:+7.1f} "
          f"{roc_auc:6.3f} {eer*100:7.2f} {tar*100:7.1f}%")
print("=" * 64)


def draw_panel(ax, name, show_ylabel):
    """Draw one model's genuine/impostor score histogram with verdict box."""
    s = stats[name]
    g, i = s["genuine"], s["impostor"]
    ax.hist(i, bins=bins, weights=np.ones_like(i) / len(i), color=IMP_COLOR,
            alpha=0.55, label=f"Impostor (n={len(i):,})",
            edgecolor="white", linewidth=0.4)
    ax.hist(g, bins=bins, weights=np.ones_like(g) / len(g), color=GEN_COLOR,
            alpha=0.55, label=f"Genuine (n={len(g)})",
            edgecolor="white", linewidth=0.4)
    ax.axvline(s["i_mean"], color=IMP_COLOR, linestyle="--", linewidth=1.6)
    ax.axvline(s["g_mean"], color=GEN_COLOR, linestyle="--", linewidth=1.6)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence: same person (0–100)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    txt = (f"AUC = {s['auc']:.3f}\nEER = {s['eer']*100:.1f}%\n"
           f"$\\Delta$ = {s['g_mean']-s['i_mean']:+.1f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.6", alpha=0.9))
    ax.legend(fontsize=8, loc="upper right")
    if show_ylabel:
        ax.set_ylabel("Fraction of pairs (within class)", fontsize=10)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {name}.[pdf|png]")


# ── Figure 1a: discriminating models (full-width, 1x3) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
for ax, name in zip(axes, DISCRIMINATING, strict=False):
    draw_panel(ax, name, show_ylabel=(ax is axes[0]))
save(fig, "fig_score_dist_strong")

# ── Figure 1b: failed models (landscape 1x2, so it sits inline with text) ─────
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
for ax, name in zip(axes, FAILED, strict=False):
    draw_panel(ax, name, show_ylabel=(ax is axes[0]))
save(fig, "fig_score_dist_failed")

# ── Figure 1c: COMBINED 5-panel (used in the LNCS/ECCV build to save space) ───
fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.4), sharey=True)
for ax, name in zip(axes, DISCRIMINATING + FAILED, strict=False):
    draw_panel(ax, name, show_ylabel=(ax is axes[0]))
save(fig, "fig_score_dist_all")

# ── Figure 2: ROC curves ──────────────────────────────────────────────────────
COLORS = ["#1565C0", "#6A1B9A", "#2E7D32", "#C62828", "#EF6C00"]
STYLES = ["-", "-", "-.", "--", ":"]

fig, ax = plt.subplots(figsize=(6, 5.5))

for (name, fpr, tpr, roc_auc), color, style in zip(roc_data, COLORS, STYLES, strict=False):
    # true ROC is a staircase between discrete operating points
    ax.step(fpr, tpr, where="post", color=color, linestyle=style,
            linewidth=2, label=f"{name}  (AUC = {roc_auc:.3f})")

ax.axvline(0.001, color="gray", linestyle=":", linewidth=1.2, label="FAR = 0.1%")
ax.plot([0, 1], [0, 1], color="lightgray", linestyle="--", linewidth=1)
# pad limits past [0, 1] so curve segments lying exactly on TAR=0/TAR=1/FAR=0
# sit inside the plot area instead of being hidden under the axis spines
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("False Accept Rate (FAR)", fontsize=11)
ax.set_ylabel("True Accept Rate (TAR)", fontsize=11)
ax.set_title("ROC Curves — Similarity-Scoring Prompt", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
ax.grid(alpha=0.3)

# inset: low-FAR operating region (placed in the empty upper-center band,
# clear of both the curves and the legend)
axins = ax.inset_axes([0.165, 0.46, 0.42, 0.42])
for (_name, fpr, tpr, _roc_auc), color, style in zip(roc_data, COLORS, STYLES, strict=False):
    axins.step(fpr, tpr, where="post", color=color, linestyle=style, linewidth=1.6)
axins.axvline(0.001, color="gray", linestyle=":", linewidth=1)
axins.set_xlim(-0.0015, 0.05)
axins.set_ylim(-0.03, 1.03)
axins.set_xlabel("FAR", fontsize=8)
axins.set_ylabel("TAR", fontsize=8)
axins.set_title("Low-FAR region (FAR $\\leq$ 0.05)", fontsize=8)
axins.tick_params(labelsize=7)
axins.grid(alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT, f"fig_roc_curves.{ext}"),
                bbox_inches="tight", dpi=300)
plt.close()
print("Saved: fig_roc_curves.[pdf|png]")
print("\nDone. Figures written to paper/manuscript/figures/")
