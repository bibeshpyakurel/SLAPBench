"""
Demographic fairness analysis for SLAPBench similarity-scoring results.

For each discriminating model and each demographic subgroup we compute the
per-subgroup verification performance:
  - genuine pairs whose subject belongs to the subgroup,
  - impostor pairs whose *both* subjects belong to the subgroup (within-group),
and report genuine/impostor mean confidence, separation, EER and AUC on that
subset. This mirrors the standard subgroup biometric protocol and exposes
whether discrimination quality is uneven across demographic groups.

Outputs:
  - console table,
  - paper/manuscript/figures/fig_fairness.{pdf,png}  (AUC by subgroup, per model)
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "paper", "manuscript", "figures")
os.makedirs(OUT, exist_ok=True)

# similarity-scoring CSVs for the three discriminating models
MODELS = [
    ("Qwen3-VL-8B",    "qwen3vl"),
    ("Claude Opus 4.8", "anthropic"),
    ("Gemma-3-12B",    "gemma3"),
]


def find_ss(model_dir):
    pats = [os.path.join(BASE, "results", "sd302b", model_dir, "latest",
                         "task8_*_similarity_score_*.csv"),
            os.path.join(BASE, "results", "sd302b", model_dir, "previous", "*",
                         "task8_*_similarity_score_*.csv")]
    return sorted(p for pat in pats for p in glob.glob(pat))[-1]


# ── demographics ──────────────────────────────────────────────────────────────
part = pd.read_csv(os.path.join(BASE, "datasets", "sd302b", "participants.csv"),
                   dtype={"id": str})
part["sid"] = part["id"].str.lstrip("0").astype(int)
part["age_band"] = pd.cut(part["age"], [0, 30, 45, 200],
                          labels=["18–30", "31–45", "46+"])
demo = part.set_index("sid")

# subgroups worth reporting (others too small among the 88 clean subjects)
SUBGROUPS = [
    ("Gender", "gender", ["male", "female"]),
    ("Race",   "race",   ["white", "african american"]),
    ("Age",    "age_band", ["18–30", "31–45", "46+"]),
]
LABELS = {"male": "Male", "female": "Female", "white": "White",
          "african american": "Black", "18–30": "18–30",
          "31–45": "31–45", "46+": "46+"}


def eer_from_scores(genuine, impostor):
    if len(genuine) == 0 or len(impostor) == 0:
        return np.nan, np.nan
    y = np.r_[np.ones_like(genuine), np.zeros_like(impostor)]
    s = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return auc(fpr, tpr), (fpr[idx] + fnr[idx]) / 2.0


def attr(sid, col):
    try:
        return demo.loc[sid, col]
    except KeyError:
        return None


# ── compute ───────────────────────────────────────────────────────────────────
results = {}  # (model, dim, group) -> dict
for mname, mdir in MODELS:
    df = pd.read_csv(find_ss(mdir))
    df["s1"] = df["subject1"].astype(int)
    df["s2"] = df["subject2"].astype(int)
    gen = df[df.label == "genuine"].copy()
    imp = df[df.label == "impostor"].copy()
    for dim, col, groups in SUBGROUPS:
        gen_a = gen["s1"].map(lambda s, col=col: attr(s, col))
        imp_a1 = imp["s1"].map(lambda s, col=col: attr(s, col))
        imp_a2 = imp["s2"].map(lambda s, col=col: attr(s, col))
        for g in groups:
            g_scores = gen.loc[gen_a == g, "similarity_score"].astype(float).values
            i_scores = imp.loc[(imp_a1 == g) & (imp_a2 == g),
                               "similarity_score"].astype(float).values
            a, e = eer_from_scores(g_scores, i_scores)
            results[(mname, dim, g)] = dict(
                gmean=g_scores.mean() if len(g_scores) else np.nan,
                imean=i_scores.mean() if len(i_scores) else np.nan,
                auc=a, eer=e, ng=len(g_scores), ni=len(i_scores))

# ── console report ────────────────────────────────────────────────────────────
print("=" * 78)
print(f"{'Model':16s} {'Dim':7s} {'Group':8s} {'nGen':>5s} {'nImp':>6s} "
      f"{'Gen':>6s} {'Imp':>6s} {'Δ':>6s} {'AUC':>6s} {'EER%':>6s}")
print("=" * 78)
for mname, _ in MODELS:
    for dim, _col, groups in SUBGROUPS:
        for g in groups:
            r = results[(mname, dim, g)]
            d = r["gmean"] - r["imean"]
            print(f"{mname:16s} {dim:7s} {LABELS[g]:8s} {r['ng']:5d} {r['ni']:6d} "
                  f"{r['gmean']:6.1f} {r['imean']:6.1f} {d:+6.1f} "
                  f"{r['auc']:6.3f} {r['eer']*100:6.1f}")
    print("-" * 78)

# ── figure: AUC by subgroup, grouped by model (gender + race) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
COLORS = {"Qwen3-VL-8B": "#1565C0", "Claude Opus 4.8": "#6A1B9A",
          "Gemma-3-12B": "#2E7D32"}

for ax, (dim, _col, groups) in zip(axes, SUBGROUPS[:2], strict=False):
    x = np.arange(len(groups))
    w = 0.26
    for k, (mname, _) in enumerate(MODELS):
        vals = [results[(mname, dim, g)]["auc"] for g in groups]
        ax.bar(x + (k - 1) * w, vals, w, label=mname, color=COLORS[mname],
               alpha=0.85, edgecolor="white")
        for xi, v in zip(x + (k - 1) * w, vals, strict=False):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups])
    ax.set_title(f"AUC by {dim}", fontsize=11, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("AUC" if ax is axes[0] else "")
    ax.grid(axis="y", alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=9, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.03), frameon=False)
fig.tight_layout(rect=[0, 0.04, 1, 1])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_fairness.{ext}"), bbox_inches="tight", dpi=300)
plt.close(fig)
print("\nSaved: fig_fairness.[pdf|png]")
