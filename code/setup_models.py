"""
SLAPBench Model Setup
=====================
Checks GPU/disk/packages and optionally downloads all three active models:
  Qwen3-VL-8B-Instruct       (Qwen/Qwen3-VL-8B-Instruct,          ~16 GB, 4-bit NF4)
  InternVL3-8B               (OpenGVLab/InternVL3-8B,              ~16 GB, bfloat16)
  Qwen2.5-VL-7B-Instruct     (Qwen/Qwen2.5-VL-7B-Instruct,        ~17 GB, 4-bit NF4)

Models are loaded directly via transformers — no vLLM required.

Usage:
    python code/setup_models.py                  # check + show instructions
    python code/setup_models.py --download       # also download all models
    python code/setup_models.py --check-only     # GPU/disk/package check only
"""

import argparse
from pathlib import Path

MODELS_DIR = Path("/media/bibesh/DATA/models")

MODELS = [
    {
        "name":    "Qwen3-VL-8B-Instruct",
        "hf_id":  "Qwen/Qwen3-VL-8B-Instruct",
        "size_gb": 16,
        "notes":  "4-bit NF4 bitsandbytes, ~8 GB VRAM at inference",
    },
    {
        "name":    "InternVL3-8B",
        "hf_id":  "OpenGVLab/InternVL3-8B",
        "size_gb": 16,
        "notes":  "bfloat16, ~16 GB VRAM",
    },
    {
        "name":    "Qwen2.5-VL-7B-Instruct",
        "hf_id":  "Qwen/Qwen2.5-VL-7B-Instruct",
        "size_gb": 17,
        "notes":  "4-bit NF4 bitsandbytes, ~6 GB VRAM at inference",
    },
]


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

        if total_vram >= 16:
            print("  ✓  Sufficient for all models (run one at a time).")
        elif total_vram >= 8:
            print("  ✓  Sufficient for Qwen3-VL and Qwen2.5-VL at 4-bit.")
            print("     InternVL3-8B requires ~16 GB bfloat16.")
        else:
            print("  ✗  Less than 8 GB VRAM — insufficient for these models.")
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
        print("\n  Install all missing packages with:")
        print("  pip install 'transformers>=4.51,<5.0' huggingface_hub accelerate "
              "bitsandbytes qwen-vl-utils einops timm pandas Pillow")
    return all_ok


# ── Disk space check ──────────────────────────────────────────────────────────

def check_disk():
    print("\n" + "=" * 60)
    print("DISK SPACE CHECK")
    print("=" * 60)
    import shutil
    total_needed = sum(m["size_gb"] for m in MODELS) + 20  # +20 GB buffer
    check_path = MODELS_DIR if MODELS_DIR.exists() else Path.home()
    stat = shutil.disk_usage(check_path)
    free_gb = stat.free / 1e9
    print(f"  Checking: {check_path}")
    print(f"  Free disk space: {free_gb:.0f} GB")
    print(f"  Space needed:    ~{total_needed} GB (all models + buffer)")
    if free_gb >= total_needed:
        print("  ✓  Sufficient disk space.")
    else:
        print(f"  ✗  Need ~{total_needed - free_gb:.0f} GB more disk space.")


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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        local_dir = MODELS_DIR / m["name"].lower().replace("-instruct", "").replace(".", "").replace("-", "")
        print(f"\n  Downloading {m['name']}  (~{m['size_gb']} GB) ...")
        print(f"  HuggingFace ID: {m['hf_id']}")
        try:
            path = snapshot_download(repo_id=m["hf_id"], repo_type="model",
                                     local_dir=str(local_dir))
            print(f"  ✓  Saved to: {path}")
        except Exception as e:
            print(f"  ✗  Failed: {e}")
            print("     If authentication is required: hf login")


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
        --resume results/internvl3/latest/task8_internvl3_zero_shot_YYYYMMDD_HHMM.csv
""")


# ── HuggingFace login reminder ────────────────────────────────────────────────

def print_hf_note():
    print("\n" + "=" * 60)
    print("HUGGINGFACE AUTHENTICATION")
    print("=" * 60)
    print("""
  All models are public but require a HuggingFace account token to download.

    1. Create a free account at https://huggingface.co
    2. Go to https://huggingface.co/settings/tokens
    3. Create a token with "read" permissions
    4. Run:  hf login
             (paste your token when prompted)

  Tokens are cached locally — you only need to do this once per machine.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download",   action="store_true",
                        help="Download all models from HuggingFace")
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
    pip install 'transformers>=4.51,<5.0' huggingface_hub accelerate \\
                bitsandbytes qwen-vl-utils einops timm pandas Pillow

  Step 2 — Log in to HuggingFace (once):
    hf login

  Step 3 — Verify system + download models:
    python code/setup_models.py --check-only
    python code/setup_models.py --download

  Step 4 — Dry run (confirms pairs load correctly, no GPU needed):
    python code/run_verification.py --dry-run --all-impostors

  Step 5 — Run all experiments (one model at a time):
    bash run_overnight.sh

  Results saved to results/<model>/latest/ after every pair (safe to interrupt and resume).
""")
