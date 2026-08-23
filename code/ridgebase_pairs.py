"""
Build mated / non-mated verification pairs from RidgeBase (Task 2, four-finger).

RidgeBase gives genuinely independent captures per subject, which fixes the
near-duplicate flaw that sank the SD302b protocol. Contact images carry a
session prefix (1_/2_), so three mated protocols are available:

  C2C   : contact session 1 vs session 2, same (subject, hand)   [pure slap]
  C2CL  : contact vs contactless, same (subject, hand)           [cross-sensor]
  CL2CL : contactless vs contactless, same (subject, hand)
  non-mated: different subjects, same hand, same protocol.

Filename conventions (Task 2):
  Contactbased/<subj>/<sess>_<subj>_<Left|Right>_Four_Fingers.bmp
  Contactless/<subj>/<sess>_<Device>_<subj>_<idx>_<LEFT|RIGHT>_image_fingerprint<code>.png

Usage:
  python code/ridgebase_pairs.py --root datasets/RidgeBase_extracted/Fingerprint_Train_Test_Split \
      --split Test --protocol c2c --out results_ridgebase/pairs_c2c_test.csv
"""
import argparse
import csv
import glob
import itertools
import os
import re

HAND_RE = re.compile(r"(left|right)", re.I)
SUBJ_RE = re.compile(r"(\d{4,})")          # subject id = first >=4-digit run
SESS_RE = re.compile(r"^(\d+)_")           # leading "<session>_" prefix


def parse_image(path):
    """Return (subject_id, hand, session) or None."""
    name = os.path.basename(path)
    h = HAND_RE.search(name)
    s = SUBJ_RE.search(name)
    if not (h and s):
        parent = os.path.basename(os.path.dirname(path))
        if parent.isdigit():
            s = re.match(r"(\d+)", parent)
    if not (h and s):
        return None
    sess = SESS_RE.match(name)
    return s.group(1), h.group(1).lower(), (sess.group(1) if sess else "0")


def collect(root, split, modality):
    """Return {(subject, hand): [(path, session)]} for one modality."""
    base = os.path.join(root, "Task2", split, modality)
    paths = []
    for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg"):
        paths += glob.glob(os.path.join(base, "**", ext), recursive=True)
    index = {}
    for p in sorted(paths):
        if p.lower().endswith(".wsq"):
            continue
        parsed = parse_image(p)
        if parsed is None:
            continue
        subj, hand, sess = parsed
        index.setdefault((subj, hand), []).append((p, sess))
    return index


def build(root, split, protocol, max_impostor_per_hand):
    cb = collect(root, split, "Contactbased") if protocol in ("c2c", "c2cl") else {}
    cl = collect(root, split, "Contactless") if protocol in ("c2cl", "cl2cl") else {}
    gallery = cb if protocol in ("c2c", "c2cl") else cl
    probe = cb if protocol == "c2c" else cl

    rows = []
    keys = sorted(set(gallery) & set(probe))

    def mated_pairs(subj, hand):
        g, p = gallery[(subj, hand)], probe[(subj, hand)]
        if protocol == "c2c":
            # different contact sessions only
            return [(a, b) for (a, sa) in g for (b, sb) in p if sa != sb and a < b]
        if protocol == "cl2cl":
            return [(a, b) for (a, _), (b, _) in itertools.combinations(g, 2)]
        return [(a, b) for (a, _) in g for (b, _) in p]  # c2cl

    for (subj, hand) in keys:
        for a, b in mated_pairs(subj, hand):
            rows.append(dict(pair_id=f"M_{subj}_{hand}_{len(rows)}", label="mated",
                             subject1=subj, subject2=subj, hand=hand,
                             img1_path=a, img2_path=b))

    # non-mated: different subjects, same hand (one capture from each pool)
    import random
    random.seed(42)
    by_hand = {}
    for (subj, hand) in keys:
        by_hand.setdefault(hand, []).append(subj)
    for hand, subs in by_hand.items():
        count = 0
        for s1, s2 in itertools.combinations(sorted(subs), 2):
            a = gallery[(s1, hand)][0][0]
            b = probe[(s2, hand)][0][0]
            rows.append(dict(pair_id=f"N_{s1}_{s2}_{hand}", label="non-mated",
                             subject1=s1, subject2=s2, hand=hand,
                             img1_path=a, img2_path=b))
            count += 1
            if max_impostor_per_hand and count >= max_impostor_per_hand:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="RidgeBase extracted root")
    ap.add_argument("--split", default="Test", choices=["Test", "Train"])
    ap.add_argument("--protocol", default="c2c", choices=["c2c", "c2cl", "cl2cl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-impostor-per-hand", type=int, default=None)
    args = ap.parse_args()

    rows = build(args.root, args.split, args.protocol, args.max_impostor_per_hand)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "label", "subject1",
                                          "subject2", "hand", "img1_path", "img2_path"])
        w.writeheader()
        w.writerows(rows)
    n_m = sum(r["label"] == "mated" for r in rows)
    n_n = len(rows) - n_m
    subs = len({r["subject1"] for r in rows} | {r["subject2"] for r in rows})
    print(f"[{args.protocol} / {args.split}] subjects={subs}  "
          f"mated={n_m}  non-mated={n_n}  total={len(rows)}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
