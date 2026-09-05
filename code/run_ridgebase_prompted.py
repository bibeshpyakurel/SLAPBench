"""
Prompted verification on RidgeBase (Task2, four-finger contactless / CL2CL).

run_verification.py builds its pairs from SD302b's slap_images.csv and is wired
to that dataset. This runner keeps the *inference* identical (same model
backends, same PROMPTS, same parsers) but feeds it a RidgeBase pair CSV built by
ridgebase_pairs_full.py, and writes results in the same shape as the SD302b runs
so downstream analysis/figures match.

Loads each model ONCE and runs all requested strategies over it (model load is
the slow part). Checkpoints after every pair; safe to interrupt and --resume.

Output columns mirror the SD302b runs, plus RidgeBase-specific `category` and
`device_pair` so AUC/EER can be sliced per comparison type at analysis time.

Usage:
  python code/run_ridgebase_prompted.py --model qwen3vl \
      --pairs results_ridgebase/pairs_ridgebase_eval.csv          # all 3 strategies
  python code/run_ridgebase_prompted.py --model qwen3vl \
      --pairs ... --prompting similarity_score --limit 20         # one strategy, smoke
"""
import argparse, csv, os, sys, time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_verification as rv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RB_RESULTS = PROJECT_ROOT / "results" / "ridgebase"

HF_REPO = {
    "qwen3vl":   "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-8B",
    "qwen25vl":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma3":    "google/gemma-3-12b-it",
}


def out_columns(score_mode):
    common = ["pair_id", "label", "category", "device_pair",
              "subject1", "subject2", "hand1", "hand2", "ground_truth"]
    tail = ["raw_response", "latency_s", "model", "prompting_setting", "timestamp"]
    return common + (["similarity_score"] if score_mode else ["llm_answer", "correct"]) + tail


def run_one(model, proc, backend, pairs, model_key, prompting, out_path, resume):
    score_mode = prompting in rv.SCORE_PROMPTS
    prompt_txt = rv.PROMPTS[prompting]
    cols = out_columns(score_mode)

    processed = set()
    if resume and out_path.exists():
        processed = set(pd.read_csv(out_path)["pair_id"].tolist())
        print(f"  resume: {len(processed)} already done")
    # Crash resilience: the sidecar .attempt file names the pair currently being
    # processed. A hard crash (segfault/OOM) leaves it pointing at the culprit;
    # on restart we skip that pair so the run makes progress instead of looping.
    attempt_path = out_path.with_suffix(out_path.suffix + ".attempt")
    skip = set()
    if attempt_path.exists():
        crashed = attempt_path.read_text().strip()
        if crashed and crashed not in processed:
            skip.add(crashed)
            with open(out_path.with_suffix(out_path.suffix + ".skipped"), "a") as sf:
                sf.write(crashed + "\n")
            print(f"  [resilience] skipping crash-inducing pair {crashed}")
    new_file = not out_path.exists() or not processed
    f = open(out_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new_file and out_path.stat().st_size == 0:
        w.writeheader()

    gt = {"mated": "A", "non-mated": "B"}
    n = 0
    for p in pairs:
        if p["pair_id"] in processed or p["pair_id"] in skip:
            continue
        attempt_path.write_text(p["pair_id"])  # mark in-flight (survives a crash)
        try:
            raw, lat = rv.call_model(model, proc, backend,
                                     p["img1_path"], p["img2_path"], prompt_txt)
        except Exception as e:
            raw, lat = f"__ERROR__ {e}", 0.0
        g = gt.get(p["label"], "B")
        rowd = {"pair_id": p["pair_id"], "label": p["label"],
                "category": p.get("category", ""), "device_pair": p.get("device_pair", ""),
                "subject1": p["subject1"], "subject2": p["subject2"],
                "hand1": p.get("hand1", ""), "hand2": p.get("hand2", ""),
                "ground_truth": g, "raw_response": raw, "latency_s": round(lat, 3),
                "model": model_key, "prompting_setting": prompting,
                "timestamp": datetime.now().isoformat(timespec="seconds")}
        if score_mode:
            rowd["similarity_score"] = rv.parse_score_response(raw)
        else:
            ans = rv.parse_answer(raw)
            rowd["llm_answer"] = ans
            rowd["correct"] = int(ans == g)
        w.writerow(rowd)
        f.flush()  # persist every pair so a hard crash loses nothing
        n += 1
        if n % 50 == 0:
            print(f"  {prompting}: {n} new pairs")
    f.close()
    if attempt_path.exists():
        attempt_path.unlink()  # clean finish -> no crash to skip next time
    print(f"  {prompting}: wrote {n} pairs -> {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=list(HF_REPO) + ["anthropic", "openai"])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--prompting", default="all",
                    choices=["all"] + list(rv.PROMPTS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--anthropic-model", default="claude-opus-4-8")
    ap.add_argument("--openai-model", default="gpt-5.6-sol")
    ap.add_argument("--results-dir", default=str(RB_RESULTS),
                    help="output tree (e.g. results/precise); default results/ridgebase")
    a = ap.parse_args()

    # Paid API backends: set the concrete model id; no local weights to load.
    if a.model == "anthropic":
        rv.MODELS["anthropic"]["api_model"] = a.anthropic_model
        rv.MODELS["anthropic"]["display"] = a.anthropic_model
    elif a.model == "openai":
        rv.MODELS["openai"]["api_model"] = a.openai_model
        rv.MODELS["openai"]["display"] = a.openai_model
    # Local open-source: models/ symlink may be unmounted -> HF cache fallback.
    elif not os.path.isdir(rv.MODELS[a.model]["hf_id"]):
        rv.MODELS[a.model]["hf_id"] = HF_REPO[a.model]

    pairs = pd.read_csv(a.pairs, dtype=str).to_dict("records")
    if a.limit:
        pairs = pairs[:a.limit]
    strategies = list(rv.PROMPTS.keys()) if a.prompting == "all" else [a.prompting]

    out_dir = Path(a.results_dir) / a.model / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"Loading {a.model} ...")
    model, proc, backend = rv.load_model(a.model)
    print(f"loaded ({backend}). {len(pairs)} pairs x {len(strategies)} strategies\n")

    for prompting in strategies:
        out_path = out_dir / f"rb_{a.model}_{prompting}_{date}.csv"
        # find an existing file to resume within this run
        if a.resume:
            prior = sorted(out_dir.glob(f"rb_{a.model}_{prompting}_*.csv"))
            if prior:
                out_path = prior[-1]
        run_one(model, proc, backend, pairs, a.model, prompting, out_path, a.resume)


if __name__ == "__main__":
    main()
