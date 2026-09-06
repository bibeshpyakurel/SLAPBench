"""
The capture-divergence gradient figure. Shows embedding-matching AUC (genuine
vs primary impostor) for each model across the three datasets, ordered by how
much genuine pairs share capture conditions:

  SD302b    genuine = same *image* (500/1000 PPI of one capture)   -> near-duplicate
  Precise   genuine = same *session* (impressions minutes apart)   -> shared appearance
  RidgeBase genuine = *independent* captures (different device/day) -> nothing shared

As that overlap decreases, AUC decays toward chance -- direct evidence the models
match capture appearance, not ridge identity. Single method (embedding cosine),
so the three points are strictly comparable.

Run:  python code/generate_gradient_figure.py
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "manuscript", "figures")
# ordered by decreasing genuine-pair overlap
DATASETS = [("sd302b", "SD302b\n(same image)"),
            ("precise", "Precise\n(same session)"),
            ("ridgebase", "RidgeBase\n(independent)")]
MODELS = [("internvl3", "InternVL3-8B", "#2E7D32", "-.", "s"),
          ("qwen25vl", "Qwen2.5-VL-7B", "#6A1B9A", "--", "^"),
          ("gemma3", "Gemma-3-12B", "#EF6C00", ":", "D")]


def embed_auc(dataset, model):
    f = os.path.join(BASE, f"results/{dataset}/{model}/embed_{model}_eval.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).dropna(subset=["cosine"])
    g = d[d.category == "primary_genuine"]["cosine"].values
    i = d[d.category == "primary_impostor"]["cosine"].values
    if len(g) == 0 or len(i) == 0:
        return None
    return roc_auc_score(np.r_[np.ones(len(g)), np.zeros(len(i))], np.r_[g, i])


def main():
    x = np.arange(len(DATASETS))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for key, name, c, ls, mk in MODELS:
        ys = [embed_auc(ds, key) for ds, _ in DATASETS]
        ax.plot(x, ys, color=c, ls=ls, marker=mk, ms=9, lw=2.2, label=name)
        for xi, yi in zip(x, ys):
            if yi is not None:
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 9), fontsize=8.5, ha="center", color=c)
    ax.axhline(0.5, color="gray", ls="--", lw=1.3)
    ax.text(len(DATASETS) - 1, 0.515, "chance", color="gray", fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DATASETS], fontsize=10)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("Embedding-matching AUC\n(genuine vs. impostor)", fontsize=11)
    ax.set_xlabel("Decreasing overlap of genuine-pair capture conditions "
                  r"$\longrightarrow$", fontsize=10.5)
    ax.set_title("As captures become independent, apparent verification\n"
                 "performance decays to chance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_capture_gradient.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("AUCs:")
    for key, name, *_ in MODELS:
        print(f"  {name}: " + "  ".join(
            f"{ds}={embed_auc(ds, key)}" for ds, _ in DATASETS))
    print("saved -> manuscript/figures/fig_capture_gradient.pdf")


if __name__ == "__main__":
    main()
