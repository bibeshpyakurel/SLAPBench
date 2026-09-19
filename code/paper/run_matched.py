"""
Run the similarity-scoring prompt on a matched-resolution manifest, reusing the
validated inference code from code/run_verification.py without modifying it.

Usage:
  python code/paper/run_matched.py --model qwen3vl --manifest results/sd302b/matched/task8_pairs_matched.csv \
      --out-dir results/sd302b/matched [--limit N] [--resume PATH]

Writes results/sd302b/matched/<model>/task8_<model>_similarity_score_matched.csv with the
same schema as the existing similarity-scoring CSVs.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_verification as rv  # noqa: E402

SS = "similarity_score"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3vl", "internvl3", "qwen25vl", "gemma3"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="results/sd302b/matched")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    pairs = pd.read_csv(args.manifest, dtype={"subject1": str, "subject2": str}).to_dict("records")
    if args.limit:
        pairs = pairs[:args.limit]

    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.resume or os.path.join(
        out_dir, f"task8_{args.model}_similarity_score_matched.csv")

    processed = set()
    if os.path.exists(out_path):
        processed = set(pd.read_csv(out_path)["pair_id"].tolist())
        print(f"Resuming — {len(processed)} already done.")

    # The registry points hf_id at a local models/ directory that may not exist
    # on this machine; fall back to the HuggingFace repo id (already cached).
    HF_REPO = {
        "qwen3vl":   "Qwen/Qwen3-VL-8B-Instruct",
        "internvl3": "OpenGVLab/InternVL3-8B",
        "qwen25vl":  "Qwen/Qwen2.5-VL-7B-Instruct",
        "gemma3":    "google/gemma-3-12b-it",
    }
    if not os.path.isdir(rv.MODELS[args.model]["hf_id"]):
        rv.MODELS[args.model]["hf_id"] = HF_REPO[args.model]
        print(f"local weights not found; using HF repo {HF_REPO[args.model]}")

    prompt_txt = rv.PROMPTS[SS]
    print(f"Loading {args.model} ...")
    model, proc, backend = rv.load_model(args.model)
    print("Model loaded.")

    fields = ["pair_id", "label", "frgp", "which_hand", "subject1", "subject2",
              "ground_truth", "similarity_score", "raw_response", "res1", "res2",
              "latency_s", "model", "prompting_setting", "timestamp"]
    write_header = not os.path.exists(out_path)
    fout = open(out_path, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        w.writeheader()

    todo = [p for p in pairs if p["pair_id"] not in processed]
    print(f"{len(todo)} pairs to run -> {out_path}")
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        try:
            raw, lat = rv.call_model(model, proc, backend,
                                     p["img1_path"], p["img2_path"], prompt_txt)
            score = rv.parse_score_response(raw)
        except Exception as e:
            raw, lat, score = f"ERROR:{e}", 0.0, None
        w.writerow({
            "pair_id": p["pair_id"], "label": p["label"], "frgp": p["frgp"],
            "which_hand": p["which_hand"], "subject1": p["subject1"],
            "subject2": p["subject2"], "ground_truth": p["ground_truth"],
            "similarity_score": score, "raw_response": str(raw)[:200],
            "res1": p["res1"], "res2": p["res2"], "latency_s": round(lat, 3),
            "model": args.model, "prompting_setting": SS,
            "timestamp": datetime.now().isoformat(timespec="seconds")})
        if i % 200 == 0:
            fout.flush()
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)}  ({rate:.1f}/s, ETA {(len(todo)-i)/rate/60:.0f} min)")
    fout.close()
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
