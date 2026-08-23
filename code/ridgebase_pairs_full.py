"""
Build the full RidgeBase (Task2, four-finger contactless / CL2CL) evaluation
pair set for v2, with every comparison category the paper needs in ONE csv.

Identity unit = (subject, hand). Contactless whole-hand photos only.

Categories (column `category`, plus a biometric `label` the runners consume):
  primary_genuine     label=mated      same subject, same hand, two *different*
                                        independent captures  (the rebuttal:
                                        genuinely separate images, not a downscale)
  primary_impostor    label=non-mated  different subjects, same hand
  diagnostic_impostor label=non-mated  same subject, *different* hand
                                        (mechanism probe: same person, different
                                        fingers -> catches identity-cue shortcutting)

Genuine pairs also carry device1/device2 and device_pair in {same,cross} so the
cross-sensor split (Apple<->Apple / google<->google vs Apple<->google) is a
groupby at analysis time, no re-run.

Two outputs:
  --out              full set   (all genuine combos; for free local open-source)
  --balanced-out     balanced   (cap genuine per (subject,hand); shared by paid
                                  models so every model scores the same pairs)

Usage:
  python code/ridgebase_pairs_full.py \
    --root datasets/ridgebase/Fingerprint_Train_Test_Split --split Test \
    --out results_ridgebase/pairs_ridgebase_full.csv \
    --balanced-out results_ridgebase/pairs_ridgebase_eval.csv
"""
import argparse, csv, glob, itertools, os, random, re, collections

SUBJ_RE = re.compile(r"(\d{4,})")
HAND_RE = re.compile(r"(LEFT|RIGHT|left|right)")
FIELDS = ["pair_id", "label", "category", "subject1", "subject2",
          "hand1", "hand2", "device1", "device2", "device_pair",
          "img1_path", "img2_path"]


def parse(path):
    n = os.path.basename(path)
    s = SUBJ_RE.search(n)
    h = HAND_RE.search(n)
    if not (s and h):
        return None
    dev = "Apple" if "Apple" in n else ("google" if "google" in n else "other")
    return s.group(1), h.group(1).lower(), dev


def collect(root, split):
    """(subject, hand) -> [(path, device)] for contactless whole-hand images."""
    base = os.path.join(root, "Task2", split, "Contactless")
    idx = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(base, "**", "*.png"), recursive=True)):
        pr = parse(p)
        if pr:
            subj, hand, dev = pr
            idx[(subj, hand)].append((p, dev))
    return idx


def row(pid, label, cat, s1, s2, h1, h2, d1, d2, p1, p2):
    return dict(pair_id=pid, label=label, category=cat, subject1=s1, subject2=s2,
                hand1=h1, hand2=h2, device1=d1, device2=d2,
                device_pair=("same" if d1 == d2 else "cross"),
                img1_path=p1, img2_path=p2)


def build(root, split, cap_genuine, cap_diag, seed):
    idx = collect(root, split)
    keys = sorted(idx)
    rng = random.Random(seed)

    genuine = []
    for (subj, hand) in keys:
        caps = idx[(subj, hand)]
        for (p1, d1), (p2, d2) in itertools.combinations(caps, 2):
            genuine.append(row(f"G_{subj}_{hand}_{len(genuine)}", "mated",
                               "primary_genuine", subj, subj, hand, hand,
                               d1, d2, p1, p2))

    # primary impostor: different subjects, same hand. Build BOTH a same-device
    # and a cross-device impostor per subject-pair, so the impostor set's device
    # composition can be matched to the genuine set (removes the "genuine is
    # cross-device, impostor is same-device" confound).
    by_hand = collections.defaultdict(list)
    for (subj, hand) in keys:
        by_hand[hand].append(subj)

    def dev_map(caps):
        d = collections.defaultdict(list)
        for p, dv in caps:
            d[dv].append(p)
        return d

    imp_same, imp_cross = [], []
    for hand, subs in by_hand.items():
        for s1, s2 in itertools.combinations(sorted(subs), 2):
            m1, m2 = dev_map(idx[(s1, hand)]), dev_map(idx[(s2, hand)])
            shared = sorted(d for d in m1 if d in m2)
            if shared:
                d = shared[0]
                imp_same.append(row(f"I_{s1}_{s2}_{hand}_s", "non-mated",
                                    "primary_impostor", s1, s2, hand, hand,
                                    d, d, m1[d][0], m2[d][0]))
            cross = [(a, b) for a in sorted(m1) for b in sorted(m2) if a != b]
            if cross:
                a, b = cross[0]
                imp_cross.append(row(f"I_{s1}_{s2}_{hand}_x", "non-mated",
                                     "primary_impostor", s1, s2, hand, hand,
                                     a, b, m1[a][0], m2[b][0]))
    impostor = imp_same + imp_cross

    # diagnostic impostor: same subject, LEFT vs RIGHT (different fingers, same person)
    diag = []
    subjects = sorted({s for (s, h) in keys})
    for subj in subjects:
        if (subj, "left") in idx and (subj, "right") in idx:
            L, R = idx[(subj, "left")], idx[(subj, "right")]
            combos = [(a, b) for a in L for b in R]
            rng.shuffle(combos)
            for (p1, d1), (p2, d2) in combos[:cap_diag]:
                diag.append(row(f"D_{subj}_{len(diag)}", "non-mated",
                               "diagnostic_impostor", subj, subj, "left", "right",
                               d1, d2, p1, p2))

    full = genuine + impostor + diag

    # balanced: cap genuine per (subject,hand), then match the impostor device
    # composition (same/cross) to the balanced genuine set so device is controlled.
    g_by_group = collections.defaultdict(list)
    for r in genuine:
        g_by_group[(r["subject1"], r["hand1"])].append(r)
    gen_bal = []
    for grp, rows in g_by_group.items():
        rng.shuffle(rows)
        gen_bal.extend(rows[:cap_genuine])
    g_same = sum(r["device_pair"] == "same" for r in gen_bal)
    g_cross = sum(r["device_pair"] == "cross" for r in gen_bal)
    rng.shuffle(imp_same); rng.shuffle(imp_cross)
    imp_bal = imp_same[:g_same] + imp_cross[:g_cross]
    balanced = gen_bal + imp_bal + diag
    rng.shuffle(full)      # mix categories so any prefix / --limit is representative
    rng.shuffle(balanced)
    return full, balanced


def write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def summary(name, rows):
    c = collections.Counter(r["category"] for r in rows)
    dp = collections.Counter(r["device_pair"] for r in rows if r["category"] == "primary_genuine")
    print(f"{name}: total={len(rows)}  " + "  ".join(f"{k}={v}" for k, v in c.items()))
    print(f"    genuine device split: same={dp['same']} cross={dp['cross']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="Test", choices=["Test", "Train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--balanced-out", required=True)
    ap.add_argument("--cap-genuine", type=int, default=12,
                    help="max genuine pairs per (subject,hand) in balanced set")
    ap.add_argument("--cap-diag", type=int, default=12,
                    help="max diagnostic impostor pairs per subject")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    full, balanced = build(a.root, a.split, a.cap_genuine, a.cap_diag, a.seed)
    write(a.out, full); write(a.balanced_out, balanced)
    summary("FULL    ", full)
    summary("BALANCED", balanced)
    print(f"saved -> {a.out}\nsaved -> {a.balanced_out}")


if __name__ == "__main__":
    main()
