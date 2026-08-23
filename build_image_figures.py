"""
Build the real-image figures for the SLAPBench paper:

  1. fig_qualitative   - two grounded example pairs (one genuine, one impostor)
                         with every model's similarity score, showing the
                         inverted/compressed failures on real SLAP images.
  2. fig_slap_anatomy  - one annotated SLAP image with the four fingers boxed
                         and labelled from the SD302b segmentation ground truth.

Source SD302b images are downsampled + contrast-stretched and copied into
manuscript/figures/ with descriptive names; the composed figure PDFs embed
those copies.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "datasets", "sd302b", "images", "baseline")
OUT = os.path.join(BASE, "manuscript", "figures")
SEG500 = os.path.join(IMG, "R", "500", "slap", "segmentation_R_500_slap_png.csv")

# locked per-model colours (consistent across every figure in the paper)
MC = {"Qwen3-VL-8B": "#1565C0", "Claude Opus 4.8": "#6A1B9A",
      "Gemma-3-12B": "#2E7D32", "InternVL3-8B": "#C62828",
      "Qwen2.5-VL-7B": "#EF6C00"}


def src(subject, res, frgp):
    return os.path.join(IMG, "R", str(res), "slap", "png",
                        f"{subject}_R_{res}_slap_{frgp}.png")


def prep(path, out_name, size=760):
    """Grayscale -> autocontrast -> downsample; save a renamed copy; return array."""
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im, cutoff=1)
    im.thumbnail((size, size), Image.LANCZOS)
    im.save(os.path.join(OUT, out_name), optimize=True)
    return np.asarray(im)


# ── example pairs (grounded in the joined model outputs) ──────────────────────
GENUINE = dict(
    subject="00002368", frgp=13, truth="Genuine — same person",
    imgs=[("00002368", 500, 13, "slap_genuine_500ppi.png"),
          ("00002368", 1000, 13, "slap_genuine_1000ppi.png")],
    captions=["Subject 2368 · 500 PPI", "Subject 2368 · 1000 PPI"],
    scores={"Qwen3-VL-8B": 100, "Claude Opus 4.8": 100, "Gemma-3-12B": 75,
            "InternVL3-8B": 9, "Qwen2.5-VL-7B": 95},
    higher_is_correct=True)

IMPOSTOR = dict(
    subject=None, frgp=13, truth="Impostor — different people",
    imgs=[("00002313", 500, 13, "slap_impostorA_500ppi.png"),
          ("00002374", 500, 13, "slap_impostorB_500ppi.png")],
    captions=["Subject 2313 · 500 PPI", "Subject 2374 · 500 PPI"],
    scores={"Qwen3-VL-8B": 65, "Claude Opus 4.8": 50, "Gemma-3-12B": 35,
            "InternVL3-8B": 85, "Qwen2.5-VL-7B": 95},
    higher_is_correct=False)


def draw_score_panel(ax, row):
    """Horizontal score bars for the five models, with correctness ticks."""
    names = list(MC.keys())
    y = np.arange(len(names))[::-1]
    # the value that would be "correct" given ground truth (for a subtle cue)
    for yi, name in zip(y, names):
        s = row["scores"][name]
        ax.barh(yi, s, height=0.6, color=MC[name], alpha=0.9, edgecolor="white")
        # model name to the LEFT of the bars (clear of the plotting area)
        ax.text(-6, yi, name, va="center", ha="right", fontsize=8.3)
        ax.text(s + 3, yi, f"{s}", va="center", ha="left", fontsize=9,
                fontweight="bold", color="0.15")
    # arrow showing which direction is "correct" for this pair
    arrow = "high $=$ correct $\\rightarrow$" if row["higher_is_correct"] \
        else "$\\leftarrow$ low $=$ correct"
    ax.text(108, -0.55, arrow, fontsize=7.5, color="0.4", ha="right",
            va="center", style="italic")
    ax.set_yticks([])
    ax.set_ylim(-0.95, len(names) - 0.3)
    ax.set_xlim(-78, 116)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Similarity score (0–100)", fontsize=8.5)
    ax.set_title("Model confidence · same person", fontsize=9,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


# ── Figure: qualitative examples (2 rows x [img, img, scores]) ─────────────────
fig = plt.figure(figsize=(9.6, 5.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35], hspace=0.32, wspace=0.18)

for r, row in enumerate([GENUINE, IMPOSTOR]):
    for c, (subj, res, frgp, fname) in enumerate(row["imgs"]):
        arr = prep(src(subj, res, frgp), fname)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(arr, cmap="gray")
        ax.set_title(row["captions"][c], fontsize=8.5)
        ax.axis("off")
    axp = fig.add_subplot(gs[r, 2])
    draw_score_panel(axp, row)
    # ground-truth tag on the left of each row
    tag = "#1565C0" if row["higher_is_correct"] else "#C62828"
    fig.text(0.012, 0.74 - r * 0.46, row["truth"], rotation=90, va="center",
             ha="center", fontsize=10, fontweight="bold", color=tag)

fig.suptitle("Real SLAP pairs with per-model similarity scores",
             fontsize=12, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0.03, 0, 1, 0.97])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_qualitative.{ext}"),
                bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved: fig_qualitative.[pdf|png]")

# ── Figure: annotated SLAP anatomy ────────────────────────────────────────────
ANATOMY = ("00002368", 500, 13, "slap_anatomy_500ppi.png")
subj, res, frgp, fname = ANATOMY
# full-res array for correct bbox scaling, then downsample factor
full = ImageOps.autocontrast(Image.open(src(subj, res, frgp)).convert("L"), 1)
W0, H0 = full.size
disp = full.copy(); disp.thumbnail((900, 900), Image.LANCZOS)
disp.save(os.path.join(OUT, fname), optimize=True)
arr = np.asarray(disp)
sx, sy = disp.size[0] / W0, disp.size[1] / H0

seg = pd.read_csv(SEG500)
rows = seg[seg.filename == f"{subj}_R_{res}_slap_{frgp}.png"]
FINGER = {2: "Index", 3: "Middle", 4: "Ring", 5: "Little"}

fig, ax = plt.subplots(figsize=(4.2, 4.6))
ax.imshow(arr, cmap="gray")
for _, rr in rows.iterrows():
    x0, y0 = rr.tlx * sx, rr.tly * sy
    w, h = (rr.trx - rr.tlx) * sx, (rr.bly - rr.tly) * sy
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="#FFB300",
                           linewidth=2.0))
    ax.text(x0 + w / 2, y0 - 8, FINGER.get(int(rr.frgp), str(rr.frgp)),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFE082", ec="#FFB300"))
ax.set_title("Four-finger SLAP with segmentation ground truth",
             fontsize=10, fontweight="bold")
ax.axis("off")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_slap_anatomy.{ext}"),
                bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved: fig_slap_anatomy.[pdf|png]")
print("Renamed source copies written to manuscript/figures/")
