"""
Pre-flight test harness for the PAID models (Claude Opus 4.8 + GPT-5.6-sol).

Purpose: spend a few cents, not a few dollars, to confirm EVERYTHING works
before the full batch:
  - API keys resolve
  - model IDs are correct (auto-discovers the exact OpenAI id)
  - the request shape is right for each API (GPT-5.x is a reasoning model:
    needs max_completion_tokens, rejects temperature)
  - responses come back and PARSE into a 0-100 score
  - real per-call token cost, so we can project the full-run bill

Runs on a handful of RidgeBase pairs (one of each category). Prints a report;
writes nothing to the results tree. Safe to run repeatedly.

Usage:
  python code/test_paid_models.py                     # 3 pairs, both models
  python code/test_paid_models.py --n 2 --only anthropic
  python code/test_paid_models.py --openai-model gpt-5.6-sol
"""
import argparse, os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_verification as rv  # SYSTEM_PROMPT, PROMPTS, image_to_base64

# $ per 1M tokens (input, output). Update if pricing changes.
PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "gpt-5.6-sol":     (5.0, 30.0),
    "gpt-4o":          (2.5, 10.0),
}
SS = rv.PROMPTS["similarity_score"]


def load_env():
    for line in open(os.path.join(os.path.dirname(__file__), "..", ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def price(model, tin, tout):
    pin, pout = PRICING.get(model, (0, 0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def pick_pairs(n):
    df = pd.read_csv("results/ridgebase/pairs_ridgebase_eval.csv", dtype=str)
    out = []
    for cat in ["primary_genuine", "primary_impostor", "diagnostic_impostor"]:
        row = df[df.category == cat].head(1)
        if len(row):
            out.append(row.iloc[0].to_dict())
    return out[:n] if n else out


# ── Anthropic (Claude Opus 4.8) ─────────────────────────────────────────────
def call_anthropic(model, i1, i2):
    import anthropic
    client = anthropic.Anthropic()
    b1, b2 = rv.image_to_base64(i1), rv.image_to_base64(i2)
    t0 = time.time()
    r = client.messages.create(
        model=model, max_tokens=50, system=rv.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": SS},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b1}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b2}},
        ]}],
    )
    txt = "".join(b.text for b in r.content if b.type == "text").strip()
    return txt, r.usage.input_tokens, r.usage.output_tokens, time.time() - t0


# ── OpenAI (GPT-5.6-sol) ────────────────────────────────────────────────────
def discover_openai_model(client, wanted):
    """Return the exact model id: `wanted` if it exists, else the closest gpt-5.6."""
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception as e:
        return wanted, f"(could not list models: {e})"
    if wanted in ids:
        return wanted, "exact match"
    cands = sorted([i for i in ids if "gpt-5.6" in i] or [i for i in ids if "gpt-5" in i])
    if cands:
        return cands[0], f"'{wanted}' not found; nearest of {cands}"
    return wanted, f"'{wanted}' not found and no gpt-5* available; ids sample: {ids[:8]}"


def call_openai(model, i1, i2):
    from openai import OpenAI
    client = OpenAI()
    b1, b2 = rv.image_to_base64(i1), rv.image_to_base64(i2)
    messages = [
        {"role": "system", "content": rv.SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": SS},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b2}"}},
        ]},
    ]
    t0 = time.time()
    # GPT-5.x reasoning model: max_completion_tokens (room for hidden reasoning
    # + the answer), reasoning_effort minimal, no temperature.
    try:
        r = client.chat.completions.create(
            model=model, messages=messages,
            max_completion_tokens=2000, reasoning_effort="minimal",
        )
    except TypeError:
        r = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=2000)
    except Exception as e:
        if "reasoning_effort" in str(e) or "unsupported" in str(e).lower():
            r = client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=2000)
        else:
            raise
    txt = (r.choices[0].message.content or "").strip()
    u = r.usage
    rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    return txt, u.prompt_tokens, u.completion_tokens, time.time() - t0, rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--only", choices=["anthropic", "openai"], default=None)
    ap.add_argument("--anthropic-model", default="claude-opus-4-8")
    ap.add_argument("--openai-model", default="gpt-5.6-sol")
    a = ap.parse_args()
    load_env()

    pairs = pick_pairs(a.n)
    print(f"Testing on {len(pairs)} RidgeBase pairs "
          f"({', '.join(p['category'] for p in pairs)})\n")
    total = 0.0

    if a.only != "openai":
        print(f"=== Claude {a.anthropic_model} (similarity-scoring) ===")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  !! ANTHROPIC_API_KEY not set\n")
        else:
            for p in pairs:
                try:
                    txt, ti, to, dt = call_anthropic(a.anthropic_model, p["img1_path"], p["img2_path"])
                    c = price(a.anthropic_model, ti, to); total += c
                    print(f"  [{p['category']:>19}] score={rv.parse_score_response(txt)!s:>4} "
                          f"raw={txt[:20]!r:22} in={ti} out={to} ${c:.5f} {dt:.1f}s")
                except Exception as e:
                    print(f"  [{p['category']}] ERROR: {type(e).__name__}: {e}")
            print()

    if a.only != "anthropic":
        print(f"=== OpenAI {a.openai_model} (similarity-scoring) ===")
        if not os.environ.get("OPENAI_API_KEY"):
            print("  !! OPENAI_API_KEY not set\n")
        else:
            from openai import OpenAI
            model, note = discover_openai_model(OpenAI(), a.openai_model)
            print(f"  model resolved -> {model}  [{note}]")
            for p in pairs:
                try:
                    txt, ti, to, dt, rt = call_openai(model, p["img1_path"], p["img2_path"])
                    c = price(model, ti, to); total += c
                    warn = "  <-- EMPTY (raise max_completion_tokens)" if not txt else ""
                    print(f"  [{p['category']:>19}] score={rv.parse_score_response(txt)!s:>4} "
                          f"raw={txt[:20]!r:22} in={ti} out={to}(reason={rt}) ${c:.5f} {dt:.1f}s{warn}")
                except Exception as e:
                    print(f"  [{p['category']}] ERROR: {type(e).__name__}: {e}")
            print()

    print(f"TOTAL TEST COST: ${total:.5f}")
    # Full-run projection: 3 strategies x 1484 pairs = 4452 calls per model per dataset
    if pairs:
        per_call = total / (len(pairs) * (2 if a.only is None else 1))
        print(f"~per-call: ${per_call:.5f}  ->  full RidgeBase run (3x1484=4452 calls/model): "
              f"~${per_call*4452:.2f}/model")


if __name__ == "__main__":
    main()
