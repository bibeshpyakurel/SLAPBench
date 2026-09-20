"""
SLAPBench Model Setup
=====================
Checks GPU/disk/packages and optionally downloads five VLMs.
Four use the local model directory; Pixtral uses the Hugging Face cache.

Models are loaded directly via transformers — no vLLM required.

Usage:
    python code/setup_models.py                  # check + show instructions
    python code/setup_models.py --download       # also download model weights
    python code/setup_models.py --check-only     # GPU/disk/package check only
"""

import argparse
from model_registry import LOCAL_MODELS, MODELS_DIR, PIXTRAL_ID


# ── GPU check ─────────────────────────────────────────────────────────────────

def check_gpu():
    print("=" * 60)
    print("GPU / VRAM CHECK")
    print("=" * 60)
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ✗  No CUDA GPU detected.")
            return False

        n = torch.cuda.device_count()
        total_vram = 0
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            total_vram += vram_gb
            print(f"  GPU {i}: {props.name}  —  {vram_gb:.0f} GB VRAM")

        print(f"\n  Total VRAM: {total_vram:.0f} GB across {n} GPU(s)")

        print("  Check each model's actual VRAM needs before inference.")
        return True

    except ImportError:
        print("  torch not installed. Run: pip install torch")
        return False


# ── Package check ─────────────────────────────────────────────────────────────

def check_packages():
    print("\n" + "=" * 60)
    print("PACKAGE CHECK")
    print("=" * 60)
    required = {
        "transformers":   "pip install 'transformers>=4.51,<5.0'",
        "huggingface_hub": "pip install huggingface_hub",
        "accelerate":     "pip install accelerate",
        "bitsandbytes":   "pip install bitsandbytes",
        "qwen_vl_utils":  "pip install qwen-vl-utils",
        "einops":         "pip install einops",
        "timm":           "pip install timm",
        "pandas":         "pip install pandas",
        "PIL":            "pip install Pillow",
    }
    all_ok = True
    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
            print(f"  ✓  {pkg}")
        except ImportError:
            print(f"  ✗  {pkg}  →  {install_cmd}")
            all_ok = False

    if not all_ok:
        print("\n  Install dependencies with: pip install -r requirements.txt")
    return all_ok


# ── Disk space check ──────────────────────────────────────────────────────────

def check_disk():
    print("\n" + "=" * 60)
    print("DISK SPACE CHECK")
    print("=" * 60)
    import shutil
    if MODELS_DIR.is_symlink() and not MODELS_DIR.exists():
        print(f"  ✗  Model symlink target is unavailable: {MODELS_DIR}")
        print("     Mount its drive or set SLAPBENCH_MODELS_DIR to available storage.")
    check_path = MODELS_DIR if MODELS_DIR.exists() else MODELS_DIR.parent
    while not check_path.exists():
        check_path = check_path.parent
    stat = shutil.disk_usage(check_path)
    free_gb = stat.free / 1e9
    print(f"  Checking: {check_path}")
    print(f"  Free disk space: {free_gb:.0f} GB")
    print("  Model sizes vary by revision; confirm free space before downloading.")
    print("  Pixtral downloads to the Hugging Face cache, possibly on another disk.")


# ── Download models ────────────────────────────────────────────────────────────

def download_models():
    print("\n" + "=" * 60)
    print("DOWNLOADING MODELS")
    print("=" * 60)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    if MODELS_DIR.is_symlink() and not MODELS_DIR.exists():
        raise SystemExit(
            f"Model symlink target is unavailable: {MODELS_DIR}. "
            "Mount its drive or set SLAPBENCH_MODELS_DIR."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for key, (repo_id, folder) in LOCAL_MODELS.items():
        local_dir = MODELS_DIR / folder
        print(f"\n  Downloading {key} from {repo_id} to {local_dir} ...")
        try:
            path = snapshot_download(repo_id=repo_id, repo_type="model",
                                     local_dir=str(local_dir))
            print(f"  ✓  Saved to: {path}")
        except Exception as e:
            print(f"  ✗  Failed: {e}")
            print("     If access requires authentication, run: hf login")
            failures.append(key)
    print(f"\n  Downloading pixtral from {PIXTRAL_ID} to the Hugging Face cache ...")
    try:
        path = snapshot_download(repo_id=PIXTRAL_ID, repo_type="model")
        print(f"  ✓  Cached at: {path}")
    except Exception as e:
        print(f"  ✗  Failed: {e}")
        failures.append("pixtral")
    if failures:
        raise SystemExit(f"Model downloads failed: {', '.join(failures)}")


# ── How to run inference ──────────────────────────────────────────────────────

def print_usage():
    print("\n" + "=" * 60)
    print("HOW TO RUN THE EVALUATION")
    print("=" * 60)
    print("""
  Models are loaded directly via transformers — no server needed.
  Use --all-impostors for the full exhaustive evaluation (7,832 pairs).

  # Dry run (no model loaded, just checks pairs):
    python code/run_verification.py --dry-run --all-impostors

  # Run Qwen3-VL-8B:
    python code/run_verification.py --model qwen3vl  --prompting zero_shot        --all-impostors --run
    python code/run_verification.py --model qwen3vl  --prompting task_description --all-impostors --run
    python code/run_verification.py --model qwen3vl  --prompting similarity_score --all-impostors --run

  # Run InternVL3-8B:
    python code/run_verification.py --model internvl3 --prompting zero_shot        --all-impostors --run
    python code/run_verification.py --model internvl3 --prompting task_description --all-impostors --run
    python code/run_verification.py --model internvl3 --prompting similarity_score --all-impostors --run

  # Run Qwen2.5-VL-7B:
    python code/run_verification.py --model qwen25vl  --prompting zero_shot        --all-impostors --run
    python code/run_verification.py --model qwen25vl  --prompting task_description --all-impostors --run
    python code/run_verification.py --model qwen25vl  --prompting similarity_score --all-impostors --run

  # Resume an interrupted run:
    python code/run_verification.py --model internvl3 --prompting zero_shot --all-impostors --run \\
        --resume results/sd302b/internvl3/latest/task8_internvl3_zero_shot_YYYYMMDD_HHMM.csv
""")


# ── HuggingFace login reminder ────────────────────────────────────────────────

def print_hf_note():
    print("\n" + "=" * 60)
    print("HUGGINGFACE AUTHENTICATION")
    print("=" * 60)
    print("""
  Some model repositories may require accepting terms or authentication.
  If access is denied, check the model page and run: hf login
  Keep tokens on this machine; never commit them to Git.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download",   action="store_true",
                        help="Download local models and cache Pixtral")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run system checks, do not download")
    args = parser.parse_args()

    check_gpu()
    check_disk()
    check_packages()
    print_hf_note()

    if args.download and not args.check_only:
        download_models()

    print_usage()

    print("\n" + "=" * 60)
    print("QUICK START SUMMARY")
    print("=" * 60)
    print("""
  Step 1 — Install packages:
    pip install -r requirements.txt

  Step 2 — Log in to HuggingFace (once):
    hf login

  Step 3 — Verify system + download models:
    python code/setup_models.py --check-only
    python code/setup_models.py --download

  Step 4 — Dry run (confirms pairs load correctly, no GPU needed):
    python code/run_verification.py --dry-run --all-impostors

  Step 5 — Run all experiments (one model at a time):
    bash code/paper/run_overnight.sh

  Results saved to results/sd302b/<model>/latest/ after every pair (safe to interrupt and resume).
""")
