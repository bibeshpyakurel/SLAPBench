"""
RidgeBase (v2) figures. Reuses the SLAPBench figure style (blue genuine / red
impostor, within-class normalized histograms, AUC/EER annotation).

Produces three figures into manuscript/figures/:
  1. fig_sd302b_vs_ridgebase : the money figure. Top row = SD302b similarity
     scores (clear separation / Qwen3-VL pinned), bottom row = RidgeBase
     similarity scores (genuine and impostor coincident, AUC ~= chance) for the
     same four open-source models. The whole rebuttal in one image.
  2. fig_ridgebase_roc       : ROC (genuine vs primary impostor) for all four
     models, all hugging the diagonal.
  3. fig_ridgebase_diagnostic: a real genuine / primary-impostor /
     diagnostic-impostor triplet with each model's score, showing the models
     cannot separate a person's own two hands.

Run:  python code/generate_ridgebase_figures.py
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "manuscript", "figures")
GEN, IMP, DIA = "#1565C0", "#C62828", "#EF6C00"
MODELS = [("qwen3vl", "Qwen3-VL-8B"), ("qwen25vl", "Qwen2.5-VL-7B"),
          ("internvl3", "InternVL3-8B"), ("gemma3", "Gemma-3-12B")]


def sd302b_scores(key):
    pats = [f"results/sd302b/{key}/latest/*similarity_score*.csv",
            f"results/sd302b/{key}/previous/*/*similarity_score*.csv"]
    f = sorted(p for pat in pats for p in glob.glob(os.path.join(BASE, pat)))[-1]
    d = pd.read_csv(f).dropna(subset=["similarity_score"])
    g = d[d.label == "genuine"]["similarity_score"].values
    i = d[d.label == "impostor"]["similarity_score"].values
    return g, i


def rb_ss(key):
    f = sorted(glob.glob(os.path.join(
        BASE, f"results/ridgebase/{key}/latest/rb_{key}_similarity_score_*.csv")))[-1]
    return pd.read_csv(f).dropna(subset=["similarity_score"])


def rb_scores(key, cat_pos="primary_genuine", cat_neg="primary_impostor"):
    d = rb_ss(key)
    g = d[d.category == cat_pos]["similarity_score"].values
    i = d[d.category == cat_neg]["similarity_score"].values
    return g, i


def auc_of(g, i):
    y = np.r_[np.ones(len(g)), np.zeros(len(i))]
    s = np.r_[g, i]
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, sk_auc(fpr, tpr)


def hist_cell(ax, g, i, title, show_ylabel):
    bins = np.linspace(0, 100, 26)
    for arr, c, lab in [(i, IMP, "Impostor"), (g, GEN, "Genuine")]:
        if len(arr):
            ax.hist(arr, bins=bins, weights=np.ones_like(arr) / len(arr),
                    color=c, alpha=0.6, label=lab, edgecolor="white", linewidth=0.3)
    if len(g): ax.axvline(g.mean(), color=GEN, ls="--", lw=1.5)
    if len(i): ax.axvline(i.mean(), color=IMP, ls="--", lw=1.5)
    _, _, a = auc_of(g, i)
    ax.text(0.04, 0.95, f"AUC={a:.2f}", transform=ax.transAxes, fontsize=9,
            va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_xlabel("Confidence (0–100)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Fraction\n(within class)", fontsize=9)


# ── Figure 1: SD302b vs RidgeBase contrast ──────────────────────────────────
def fig_contrast():
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.0), sharex=True)
    for col, (key, name) in enumerate(MODELS):
        g, i = sd302b_scores(key)
        hist_cell(axes[0, col], g, i, name, col == 0)
        g, i = rb_scores(key)
        hist_cell(axes[1, col], g, i, "", col == 0)
    axes[0, 0].annotate("SD302b\n(same-capture pairs)", xy=(-0.55, 0.5),
                        xycoords="axes fraction", fontsize=12, fontweight="bold",
                        ha="center", va="center", rotation=90, color="0.25")
    axes[1, 0].annotate("RidgeBase\n(independent captures)", xy=(-0.55, 0.5),
                        xycoords="axes fraction", fontsize=12, fontweight="bold",
                        ha="center", va="center", rotation=90, color="0.25")
    h = [plt.Rectangle((0, 0), 1, 1, color=GEN, alpha=0.6),
         plt.Rectangle((0, 0), 1, 1, color=IMP, alpha=0.6)]
    fig.legend(h, ["Genuine", "Impostor"], loc="upper center", ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0.05, 0, 1, 0.98])
    save(fig, "fig_sd302b_vs_ridgebase")


# ── Figure 2: RidgeBase ROC ─────────────────────────────────────────────────
def fig_roc():
    colors = ["#1565C0", "#6A1B9A", "#2E7D32", "#EF6C00"]
    styles = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(6, 5.6))
    for (key, name), c, st in zip(MODELS, colors, styles, strict=False):
        g, i = rb_scores(key)
        fpr, tpr, a = auc_of(g, i)
        ax.step(fpr, tpr, where="post", color=c, ls=st, lw=2,
                label=f"{name} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, label="Chance (AUC=0.5)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Accept Rate", fontsize=11)
    ax.set_ylabel("True Accept Rate", fontsize=11)
    ax.set_title("RidgeBase ROC — Genuine vs. Impostor\n(independent captures)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_ridgebase_roc")


# ── Figure 3: diagnostic triplet with per-model scores ──────────────────────
def _pick(dfkey, category):
    """Pick a representative pair id for a category (median-scored by qwen3vl)."""
    d = rb_ss(dfkey)
    sub = d[d.category == category].copy()
    sub = sub.sort_values("similarity_score")
    return sub.iloc[len(sub) // 2]["pair_id"]


def fig_diagnostic():
    from PIL import Image
    pairs = pd.read_csv(os.path.join(BASE, "results/ridgebase/pairs_ridgebase_eval.csv"),
                        dtype=str)
    # per-model score lookup by pair_id
    score = {}
    for key, _ in MODELS:
        d = rb_ss(key).set_index("pair_id")["similarity_score"]
        score[key] = d.to_dict()
    cats = [("primary_genuine", "Genuine\n(same hand, 2 captures)", GEN),
            ("primary_impostor", "Impostor\n(different people)", IMP),
            ("diagnostic_impostor", "Diagnostic impostor\n(same person, other hand)", DIA)]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10),
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for r, (cat, title, col) in enumerate(cats):
        pid = _pick("qwen3vl", cat)
        row = pairs[pairs.pair_id == pid].iloc[0]
        for c, img in enumerate([row.img1_path, row.img2_path]):
            ax = axes[r, c]
            try:
                im = Image.open(os.path.join(BASE, img)).convert("L")
                im.thumbnail((500, 500))
                ax.imshow(im, cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "image", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(col); s.set_linewidth(2.5)
        axes[r, 0].set_ylabel(title, fontsize=11, fontweight="bold", color=col)
        # scores panel
        axp = axes[r, 2]; axp.axis("off")
        names = [n for _, n in MODELS]
        vals = [score[k].get(pid, np.nan) for k, _ in MODELS]
        ypos = np.arange(len(names))[::-1]
        axp.barh(ypos, vals, color=col, alpha=0.55, height=0.62)
        for y, v, nm in zip(ypos, vals, names, strict=False):
            axp.text(2, y + 0.34, nm, va="bottom", ha="left", fontsize=8.5,
                     fontweight="bold", color="0.2")
            axp.text(v + 2, y, f"{v:.0f}", va="center", ha="left",
                     fontsize=9.5, fontweight="bold")
        axp.set_yticks([]); axp.set_xlim(0, 108)
        axp.axvline(50, color="0.55", ls=":", lw=1.2)
        axp.text(50, len(names) - 0.3, "50 = uncertain", fontsize=7.5,
                 color="0.5", ha="center")
        axp.set_title("model similarity score (0–100)" if r == 0 else "",
                      fontsize=9.5)
    fig.suptitle("A model that matches ridges would score the bottom row LOW; "
                 "instead all rows score alike.", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig_ridgebase_diagnostic")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved -> manuscript/figures/{name}.pdf")


if __name__ == "__main__":
    fig_contrast()
    fig_roc()
    fig_diagnostic()
