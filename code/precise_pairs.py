"""
Build the Precise evaluation pair set. Precise is a four-finger SLAP *livescan*
dataset (contact, the operational border modality, like SD302b) but with
MULTIPLE independent impressions per (subject, position) -- so, unlike SD302b,
genuine pairs are independent captures, not two resolutions of one image.

Filename: <subject>_<...>_<...>_<position>.jpeg  where position in {13,14,15}.
  13 = right four-finger slap, 14 = left four-finger slap, 15 = thumbs.
We use 13 and 14 (four-finger), matching the SD302b framing.

Categories (same as the RidgeBase set):
  primary_genuine     mated      same subject, same position, different impressions
  primary_impostor    non-mated  different subjects, same position
  diagnostic_impostor  non-mated  same subject, position 13 vs 14 (right vs left)

Outputs a full set and a balanced set (genuine capped per group, impostor
matched to genuine count per position, diagnostic sampled).

Usage:
  python code/precise_pairs.py --root datasets/Precise \
    --out results/precise/pairs_precise_full.csv \
    --balanced-out results/precise/pairs_precise_eval.csv
"""
import argparse, csv, glob, itertools, os, random, collections

FIELDS = ["pair_id", "label", "category", "subject1", "subject2",
          "hand1", "hand2", "device_pair", "img1_path", "img2_path"]
FOUR = {"13", "14"}


def parse(path):
    n = os.path.basename(path)
    subj = n.split("_")[0]
    pos = n.rsplit("_", 1)[1].split(".")[0]
    return subj, pos


def collect(root):
    idx = collections.defaultdict(list)          # (subject, pos) -> [paths]
    for p in sorted(glob.glob(os.path.join(root, "*.jpeg"))):
        subj, pos = parse(p)
        if pos in FOUR:
            idx[(subj, pos)].append(p)
    return idx


def row(pid, label, cat, s1, s2, h1, h2, p1, p2):
    return dict(pair_id=pid, label=label, category=cat, subject1=s1, subject2=s2,
                hand1=h1, hand2=h2, device_pair="na", img1_path=p1, img2_path=p2)


def build(root, cap_gen, cap_imp_per_pos, cap_diag, seed):
    idx = collect(root)
    rng = random.Random(seed)

    genuine = []
    for (subj, pos), imgs in idx.items():
        for a, b in itertools.combinations(imgs, 2):
            genuine.append(row(f"G_{subj}_{pos}_{len(genuine)}", "mated",
                               "primary_genuine", subj, subj, pos, pos, a, b))

    # primary impostor: different subjects, same position (one image each)
    by_pos = collections.defaultdict(list)
    for (subj, pos) in idx:
        by_pos[pos].append(subj)
    impostor = []
    for pos, subs in by_pos.items():
        subs = sorted(subs)
        for s1, s2 in itertools.combinations(subs, 2):
            impostor.append(row(f"I_{s1}_{s2}_{pos}", "non-mated",
                               "primary_impostor", s1, s2, pos, pos,
                               idx[(s1, pos)][0], idx[(s2, pos)][0]))

    # diagnostic: same subject, position 13 vs 14
    diag = []
    subjects = sorted({s for (s, p) in idx})
    for subj in subjects:
        if (subj, "13") in idx and (subj, "14") in idx:
            combos = [(a, b) for a in idx[(subj, "13")] for b in idx[(subj, "14")]]
            rng.shuffle(combos)
            for a, b in combos[:cap_diag]:
                diag.append(row(f"D_{subj}_{len(diag)}", "non-mated",
                               "diagnostic_impostor", subj, subj, "13", "14", a, b))

    full = genuine + impostor + diag

    # balanced: cap genuine per (subject,pos); match impostor count per position
    g_by = collections.defaultdict(list)
    for r in genuine:
        g_by[(r["subject1"], r["hand1"])].append(r)
    gen_bal = []
    for grp, rows in g_by.items():
        rng.shuffle(rows); gen_bal.extend(rows[:cap_gen])
    # impostors: sample per position to match the balanced genuine count per pos
    gpos = collections.Counter(r["hand1"] for r in gen_bal)
    imp_by = collections.defaultdict(list)
    for r in impostor:
        imp_by[r["hand1"]].append(r)
    imp_bal = []
    for pos, target in gpos.items():
        pool = imp_by[pos]; rng.shuffle(pool)
        imp_bal.extend(pool[:min(target, cap_imp_per_pos)])
    balanced = gen_bal + imp_bal + diag
    rng.shuffle(full); rng.shuffle(balanced)
    return full, balanced


def write(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def summary(name, rows):
    c = collections.Counter(r["category"] for r in rows)
    subs = len({r["subject1"] for r in rows} | {r["subject2"] for r in rows})
    print(f"{name}: total={len(rows)} subjects={subs}  " +
          "  ".join(f"{k}={v}" for k, v in c.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--balanced-out", required=True)
    ap.add_argument("--cap-genuine", type=int, default=3)
    ap.add_argument("--cap-imp-per-pos", type=int, default=400)
    ap.add_argument("--cap-diag", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    full, bal = build(a.root, a.cap_genuine, a.cap_imp_per_pos, a.cap_diag, a.seed)
    write(a.out, full); write(a.balanced_out, bal)
    summary("FULL    ", full); summary("BALANCED", bal)
    print(f"saved -> {a.out}\nsaved -> {a.balanced_out}")


if __name__ == "__main__":
    main()
