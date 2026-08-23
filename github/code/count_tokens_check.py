"""
Quick token count check — run this BEFORE the full experiment to get accurate cost estimates.
Uses the count_tokens endpoint (no inference, no charge).
"""

import anthropic, base64, io, os
from pathlib import Path
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).parent.parent

try:
    from dotenv import dotenv_values
    env = dotenv_values(PROJECT_ROOT / ".env")
except ImportError:
    env = {}

api_key = os.environ.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: Set ANTHROPIC_API_KEY in .env or environment")
    raise SystemExit(1)

client = anthropic.Anthropic(api_key=api_key)

PROMPT_SIMILARITY = (
    "You are a forensic fingerprint examiner analyzing two SLAP fingerprint images.\n"
    "A SLAP image captures four fingers (index, middle, ring, little) pressed simultaneously on a scanner.\n\n"
    "Your task: Examine the ridge flow patterns, ridge endings, bifurcations, and the relative shape and spacing "
    "of each finger across both images. Then return a single integer between 0 and 100 representing your "
    "confidence that both images belong to the same person, where:\n"
    "  0   = 0% confident (certain they are different people)\n"
    "  50  = 50% confident (completely uncertain)\n"
    "  100 = 100% confident (certain they are the same person)\n\n"
    "A score of 65 means you are 65% confident the images are from the same person.\n"
    "Reply with the number only. Do not include any text, explanation, or label."
)


def to_b64(path: Path, size: int = 448) -> str:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    img = canvas.resize((size, size), Image.LANCZOS).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def count(img1_b64: str, img2_b64: str, prompt: str, model: str) -> int:
    r = client.messages.count_tokens(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img1_b64}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img2_b64}},
                {"type": "text", "text": prompt},
            ]
        }]
    )
    return r.input_tokens


# Use two real images from the dataset
img_dir = PROJECT_ROOT / "dataset/sd302b/images/baseline/R/500/slap/png"
images = sorted(img_dir.glob("*.png"))
if len(images) < 2:
    print(f"ERROR: Need at least 2 images in {img_dir}")
    raise SystemExit(1)

print("Preprocessing images (448×448 PNG) ...")
img1_b64 = to_b64(images[0])
img2_b64 = to_b64(images[1])

print(f"Image 1: {images[0].name}")
print(f"Image 2: {images[1].name}")
print()

TOTAL_PAIRS = 7832
OUTPUT_TOKENS_PER_CALL = 20  # similarity score is 1-3 digits

for model, input_price, output_price in [
    ("claude-haiku-4-5",  1.00,  5.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-opus-4-8",   5.00, 25.00),
]:
    tokens = count(img1_b64, img2_b64, PROMPT_SIMILARITY, model)
    input_cost_1 = (TOTAL_PAIRS * tokens / 1_000_000) * input_price
    output_cost_1 = (TOTAL_PAIRS * OUTPUT_TOKENS_PER_CALL / 1_000_000) * output_price
    total_1 = input_cost_1 + output_cost_1

    print(f"{'='*55}")
    print(f"Model: {model}")
    print(f"  Input tokens per call: {tokens:,}")
    print(f"  1 strategy  ({TOTAL_PAIRS:,} calls): ${total_1:.2f}")
    print(f"  All 3 strat ({TOTAL_PAIRS*3:,} calls): ${total_1*3:.2f}")
    print(f"  Fits $100 budget?  1-strat: {'YES' if total_1 < 100 else 'NO'}  |  all-3: {'YES' if total_1*3 < 100 else 'NO'}")
    print()
