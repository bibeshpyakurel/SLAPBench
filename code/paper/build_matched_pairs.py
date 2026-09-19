"""
Task 2/3 — build matched-resolution pair manifests (the resolution-confound
control) and fully auditable pair CSVs.

Matched-resolution protocol (both classes share the SAME 1000-vs-500 structure):
  - Mated (genuine):  subject A 1000 PPI  vs  subject A 500 PPI, same FRGP.
  - Non-mated (impostor): subject A 1000 PPI vs subject B 500 PPI (A<B), same FRGP.

Two orderings are emitted:
  results/sd302b/matched/            image order = (1000, 500)
  results/sd302b/matched_reversed/   image order = (500, 1000)   [Task 2b ablation]

Each manifest has explicit, reproducible columns:
  pair_id, label, frgp, subject1, subject2, img1_path, img2_path, res1, res2
"""

import itertools
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLAP_DF = os.path.join(BASE, "datasets", "sd302b", "slap_images.csv")
DATASET_ROOT = os.path.join(BASE, "datasets", "sd302b")

df = pd.read_csv(SLAP_DF, dtype={"subject_id": str})
r500 = df[(df.device == "R") & (df.resolution_ppi == 500) & (~df.has_errata)]
r1000 = df[(df.device == "R") & (df.resolution_ppi == 1000) & (~df.has_errata)]


def build(reverse: bool):
    rows = []
    for frgp in [13, 14]:
        hand = "right" if frgp == 13 else "left"
        p500 = r500[r500.frgp_slap == frgp].set_index("subject_id")
        p1000 = r1000[r1000.frgp_slap == frgp].set_index("subject_id")
        subs = sorted(set(p500.index) & set(p1000.index))

        def add(pair_id, label, s1, s2, hi_row, lo_row, frgp=frgp, hand=hand):
            # hi_row = the 1000 PPI side, lo_row = the 500 PPI side (canonical)
            hi = (str(hi_row.file_path), 1000)
            lo = (str(lo_row.file_path), 500)
            a, b = (lo, hi) if reverse else (hi, lo)   # reversed => 500 first
            rows.append(dict(
                pair_id=pair_id, label=label, frgp=frgp, which_hand=hand,
                subject1=s1, subject2=s2,
                img1_path=os.path.join(DATASET_ROOT, a[0]),
                img2_path=os.path.join(DATASET_ROOT, b[0]),
                res1=a[1], res2=b[1],
                ground_truth="A" if label == "genuine" else "B"))

        # mated: A_1000 vs A_500
        for sid in subs:
            add(f"G_{sid}_frgp{frgp}", "genuine", sid, sid,
                p1000.loc[sid], p500.loc[sid])
        # non-mated: A_1000 vs B_500, A<B, all C(n,2)
        for s1, s2 in itertools.combinations(subs, 2):
            add(f"I_{s1}_{s2}_frgp{frgp}", "impostor", s1, s2,
                p1000.loc[s1], p500.loc[s2])
    return pd.DataFrame(rows)


for reverse, outdir in [(False, "results/sd302b/matched"), (True, "results/sd302b/matched_reversed")]:
    m = build(reverse)
    os.makedirs(os.path.join(BASE, outdir), exist_ok=True)
    path = os.path.join(BASE, outdir, "task8_pairs_matched.csv")
    m.to_csv(path, index=False)
    ng = (m.label == "genuine").sum()
    ni = (m.label == "impostor").sum()
    order = "500->1000" if reverse else "1000->500"
    res_pairs = m.groupby(["label"]).apply(
        lambda g: f"res1={g.res1.unique().tolist()} res2={g.res2.unique().tolist()}")
    print(f"[{outdir}] order={order}  genuine={ng} impostor={ni} total={len(m)}")
    for lab, s in res_pairs.items():
        print(f"    {lab}: {s}")
    print(f"    saved -> {path}")
