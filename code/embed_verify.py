"""
Embedding-based fingerprint verification with MLLM vision encoders.

This is the evaluation reviewer mtTv asked for: instead of prompting the model,
we extract an image embedding from each model's *vision encoder*, match a pair
by cosine similarity, and report standard biometric metrics (AUC, EER,
TAR@FAR). No language head, no prompt.

Reuses the loaders in run_verification.py. Each backend exposes its vision
tower slightly differently, so extract_embedding() has one branch per model;
run with --limit 20 first as a smoke test before a full sweep.

Usage:
  python code/embed_verify.py --model qwen3vl \
      --pairs results_ridgebase/pairs_c2cl_test.csv \
      --out results_ridgebase/embed_qwen3vl_c2cl.csv [--limit 20]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import run_verification as rv  # noqa: E402

HF_REPO = {
    "qwen3vl":   "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-8B",
    "qwen25vl":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma3":    "google/gemma-3-12b-it",
    "pixtral":   "unsloth/Pixtral-12B-2409-bnb-4bit",
}


def _pool(x):
    """Mean-pool a [.., tokens, hidden] or [tokens, hidden] tensor -> [hidden]."""
    # Some vision towers (Qwen3-VL, deepstack variants) return a tuple/list;
    # the image features are the first element.
    while isinstance(x, (tuple, list)):
        x = x[0]
    if x.dim() == 3:
        x = x.mean(dim=1).squeeze(0)
    elif x.dim() == 2:
        x = x.mean(dim=0)
    return x.float().cpu().numpy()


def extract_embedding(model, proc, backend, image_path):
    """Return an L2-normalized image embedding from the model's vision encoder."""
    import torch
    img = rv.load_pil(image_path, size=448)  # grayscale->autocontrast->448 RGB

    with torch.no_grad():
        if backend == "internvl3":
            # InternVL exposes extract_feature(pixel_values) -> [n_patch, hidden]
            pv = _internvl_pixel_values(model, img)
            feats = model.extract_feature(pv)
            emb = _pool(feats)

        elif backend in ("qwen25vl", "qwen3vl"):
            # Qwen-VL: processor -> pixel_values + image_grid_thw; model.visual(...)
            inputs = proc(images=img, text="", return_tensors="pt")
            pv = inputs["pixel_values"].to(model.device, dtype=next(model.parameters()).dtype)
            grid = inputs["image_grid_thw"].to(model.device)
            feats = model.visual(pv, grid_thw=grid)
            emb = _pool(feats)

        elif backend == "gemma3":
            # Gemma3 (SigLIP tower): get_image_features -> [n_patch, hidden]
            inputs = proc(images=img, text="<start_of_image>", return_tensors="pt").to(model.device)
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            emb = _pool(feats if not isinstance(feats, (list, tuple)) else feats[0])

        elif backend == "pixtral":
            # Pixtral vision tower (via Llava wrapper): get_image_features(pixel_values,
            # image_sizes) -> [n_images, n_patch, hidden], post multimodal projection.
            inputs = proc(images=img, text="", return_tensors="pt")
            pv = inputs["pixel_values"].to(model.device, dtype=next(model.parameters()).dtype)
            sizes = inputs["image_sizes"].to(model.device)
            feats = model.get_image_features(pixel_values=pv, image_sizes=sizes)
            emb = _pool(feats)
        else:
            raise ValueError(f"no embedding path for backend {backend}")

    emb = emb.astype(np.float32)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def _internvl_pixel_values(model, img):
    """Build InternVL pixel_values tensor from a PIL image."""
    import torchvision.transforms as T
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize(mean, std)])
    pv = tf(img.convert("RGB")).unsqueeze(0)
    return pv.to(model.device, dtype=next(model.parameters()).dtype)


def eer_auc_tar(mated, nonmated):
    from sklearn.metrics import roc_curve, auc
    y = np.r_[np.ones_like(mated), np.zeros_like(nonmated)]
    s = np.r_[mated, nonmated]
    fpr, tpr, _ = roc_curve(y, s)
    a = auc(fpr, tpr)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[k] + fnr[k]) / 2
    tar = max([np.mean(mated >= t) for t in np.unique(s) if np.mean(nonmated >= t) <= 0.001] + [0])
    return a, eer, tar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HF_REPO))
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip pairs already in --out (recover from a crash)")
    args = ap.parse_args()

    if not os.path.isdir(rv.MODELS[args.model]["hf_id"]):
        rv.MODELS[args.model]["hf_id"] = HF_REPO[args.model]

    pairs = pd.read_csv(args.pairs).to_dict("records")
    if args.limit:
        pairs = pairs[:args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        try:
            done = set(pd.read_csv(args.out)["pair_id"].tolist())
            print(f"resume: {len(done)} pairs already embedded")
        except Exception:
            done = set()
    pairs = [p for p in pairs if p["pair_id"] not in done]

    print(f"Loading {args.model} ...")
    model, proc, backend = rv.load_model(args.model)
    print("loaded.")

    cache = {}
    def emb(path):
        if path not in cache:
            cache[path] = extract_embedding(model, proc, backend, path)
        return cache[path]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    append = args.resume and done
    fout = open(args.out, "a" if append else "w", newline="")
    w = csv.DictWriter(fout, fieldnames=["pair_id", "label", "category",
                                         "device_pair", "cosine", "latency_s"])
    if not append:
        w.writeheader()
    t0 = time.time()
    for i, p in enumerate(pairs, 1):
        try:
            e1, e2 = emb(p["img1_path"]), emb(p["img2_path"])
            cos = float(np.dot(e1, e2))
        except Exception as ex:
            cos = float("nan")
            if i <= 3:
                print(f"  [warn] {ex}")
        w.writerow({"pair_id": p["pair_id"], "label": p["label"],
                    "category": p.get("category", ""), "device_pair": p.get("device_pair", ""),
                    "cosine": round(cos, 6), "latency_s": round(time.time() - t0, 2)})
        if i % 200 == 0:
            fout.flush(); print(f"  {i}/{len(pairs)}")
    fout.close()

    df = pd.read_csv(args.out).dropna(subset=["cosine"])
    m = df[df.label == "mated"]["cosine"].values
    n = df[df.label == "non-mated"]["cosine"].values
    if len(m) and len(n):
        a, eer, tar = eer_auc_tar(m, n)
        print(f"\n{args.model} EMBEDDING verification ({len(df)} pairs)")
        print(f"  mated_mean={m.mean():.3f}  nonmated_mean={n.mean():.3f}")
        print(f"  AUC={a:.4f}  EER={eer*100:.2f}%  TAR@FAR=0.1%={tar*100:.1f}%")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
