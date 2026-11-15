"""Local, fixed-pair InternVL3-38B inference and synthetic capacity checks.

No downloads, training, pair generation, implicit retry, or device fallback.
Heavy dependencies and upstream custom code are imported only by explicit calls.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path, PureWindowsPath
import re
import struct
import subprocess
import sys
import time

GIB = 1024 ** 3
REPO_ID = "OpenGVLab/InternVL3-38B"
REVISION = "b99bcba4fe0c51d9e475af25ec000dbfea6284af"
CURRENT_BUNDLE_SHA256 = "48ec4b7652940aef16f3eb8a843b5f9825752219a75f7280b461be63efe2ac86"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ("code/internvl38b_runtime.py", "code/run_precise_internvl38b.py",
                "requirements-internvl38b.txt")


class PreflightError(ValueError):
    """An explicit precondition failed; no fallback is allowed."""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Config:
    schema_version: int
    repo_id: str
    revision: str
    transformers_version: str
    gpu_weight_budget_gib: int
    gpu_reserve_gib: int
    cpu_reserve_gib: int
    cpu_dtype: str
    cpu_threads: int
    compute_dtype: str
    quantization: str
    attention: str
    image_size: int
    max_tiles: int
    thumbnail: bool
    max_new_tokens: int
    seed: int
    system_prompt: str
    user_prompt: str

    def validate(self):
        integers = (self.schema_version, self.gpu_weight_budget_gib, self.gpu_reserve_gib,
                    self.cpu_reserve_gib, self.image_size, self.max_tiles, self.max_new_tokens,
                    self.cpu_threads, self.seed)
        if (any(type(x) is not int for x in integers) or self.schema_version != 1
                or min(integers[1:-1]) < 1 or not 0 <= self.seed < 2**32
                or self.repo_id != REPO_ID or self.revision != REVISION
                or self.transformers_version != "4.48.3" or self.image_size != 448
                or not 1 <= self.max_tiles <= 12 or self.max_new_tokens > 128
                or self.cpu_dtype != "bfloat16" or self.compute_dtype != "bfloat16"
                or self.quantization != "nf4_double" or self.attention != "eager"
                or type(self.thumbnail) is not bool):
            raise PreflightError("Unsupported explicit InternVL3-38B configuration.")
        for prompt in (self.system_prompt, self.user_prompt):
            if not isinstance(prompt, str) or not prompt.strip() or any(
                    marker in prompt for marker in ("<image>", "<IMG_CONTEXT>", "<|im_")):
                raise PreflightError("Explicit text-only system/user prompts are required.")


def load_config(path):
    config = Config(**json.loads(Path(path).read_text()))
    config.validate()
    return config


def verify_snapshot(directory, receipt_path, *, full=False):
    """Verify custom source before import; optionally hash every weight byte."""
    directory = Path(directory).resolve(strict=True)
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("repo_id") != REPO_ID or receipt.get("revision") != REVISION:
        raise PreflightError("The model receipt does not identify the pinned official snapshot.")
    seen, total = set(), 0
    for row in receipt["files"]:
        name = row["file"]
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory) or name in seen or not path.is_file():
            raise PreflightError("Unsafe, duplicate or missing snapshot member.")
        seen.add(name)
        total += path.stat().st_size
        if path.stat().st_size != row["bytes"] or (
                (full or path.suffix != ".safetensors") and digest(path) != row["sha256"]):
            raise PreflightError("Snapshot size or digest mismatch; original assets must stay unchanged.")
    if total != receipt["bytes"] or list(directory.rglob("*.incomplete")):
        raise PreflightError("Incomplete snapshot.")
    return {"receipt_sha256": digest(receipt_path), "files": len(seen), "bytes": total,
            "verification": "full_sha256" if full else "all_sizes_and_nonweight_sha256"}


def tensor_inventory(directory):
    """Read Safetensors headers only, never allocate checkpoint tensors."""
    directory = Path(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    specs = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = (directory / shard).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()):
            raise PreflightError("Weight index escapes snapshot.")
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise PreflightError("Truncated Safetensors header.")
            length = struct.unpack("<Q", length_bytes)[0]
            if not 0 < length <= min(16 * 1024 * 1024, path.stat().st_size - 8):
                raise PreflightError("Invalid Safetensors header length.")
            header = json.loads(stream.read(length))
        offsets = []
        for name, row in header.items():
            if name == "__metadata__":
                continue
            shape, (start, end) = row["shape"], row["data_offsets"]
            if (name in specs or index["weight_map"].get(name) != shard or row["dtype"] != "BF16"
                    or not shape or any(type(n) is not int or n < 1 for n in shape)
                    or end - start != 2 * math.prod(shape)):
                raise PreflightError("Unsupported, repeated or inconsistent checkpoint tensor.")
            specs[name] = {"shape": shape, "elements": math.prod(shape)}
            offsets.append((start, end))
        offsets.sort()
        if (not offsets or offsets[0][0] != 0 or offsets[-1][1] != path.stat().st_size - 8 - length
                or any(a[1] != b[0] for a, b in zip(offsets, offsets[1:], strict=False))):
            raise PreflightError("Invalid Safetensors payload boundaries.")
    if set(specs) != set(index["weight_map"]) or sum(
            row["elements"] * 2 for row in specs.values()) != index["metadata"]["total_size"]:
        raise PreflightError("Incomplete checkpoint index.")
    return specs


def runtime_identity():
    import importlib.metadata
    import torch
    packages = {key: importlib.metadata.version(key) for key in (
        "torch", "torchvision", "transformers", "tokenizers", "huggingface-hub", "accelerate",
        "bitsandbytes", "timm", "einops", "safetensors", "Pillow", "requests")}
    if sys.version_info[:2] != (3, 12) or packages["transformers"] != "4.48.3":
        raise PreflightError("Use the dedicated Python 3.12 / Transformers 4.48.3 environment.")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", *SOURCE_PATHS],
                                     cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1", "--", *SOURCE_PATHS],
                                    cwd=ROOT, text=True).strip()
    return {"python": sys.version, "executable_sha256": digest(Path(sys.executable).resolve()),
            "packages": packages, "source_sha256": {p: digest(ROOT / p) for p in SOURCE_PATHS},
            "source_commit": commit or None, "protected_source_clean": not dirty,
            "kernel": platform.release(), "torch_build": torch.__version__,
            "torch_cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def model_skeleton(directory, config):
    """Execute verified local upstream code on meta, with no real model weights."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    cfg = AutoConfig.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    if (cfg.force_image_size != config.image_size or cfg.llm_config.num_hidden_layers != 64
            or cfg.template != "internvl2_5"):
        raise PreflightError("Unexpected pinned model architecture.")
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True,
                                             local_files_only=True, use_fast=False)
    if any(tokenizer.convert_tokens_to_ids(token) is None for token in (
            "<img>", "</img>", "<IMG_CONTEXT>", "<|im_end|>")):
        raise PreflightError("Required image/conversation token is missing.")
    # Upstream computes drop-path scalars via torch.linspace(...).item().
    # Keep tiny constructor buffers on CPU; every parameter must still be meta.
    with init_empty_weights(include_buffers=False):
        model = AutoModel.from_config(cfg, trust_remote_code=True, use_flash_attn=False,
                                      torch_dtype=torch.bfloat16)
    if any(p.device.type != "meta" for p in model.parameters()):
        raise PreflightError("Architecture check unexpectedly allocated model weights.")
    linear = {name + ".weight" for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear) and name != "language_model.lm_head"}
    return model, tokenizer, linear


def atomic_group(name):
    match = re.match(r"(language_model\.model\.layers\.\d+)\.", name)
    if match:
        return match[1]
    for group in ("vision_model", "mlp1", "language_model.model.embed_tokens",
                  "language_model.lm_head", "language_model.model.norm"):
        if name.startswith(group + "."):
            return group
    raise PreflightError("Unrecognized tensor group; device placement must be reviewed.")


def quantized_size(elements):
    # NF4 bytes + nested scale storage + conservative per-tensor metadata.
    blocks = math.ceil(elements / 64)
    return math.ceil(elements / 2) + blocks + 4 * math.ceil(blocks / 256) + 4096


def memory_plan(specs, linear_weights, config, resources):
    """Fixed GPU budget; never shrink images or silently change device policy."""
    groups = defaultdict(lambda: {"gpu": 0, "cpu": 0})
    for name, row in specs.items():
        group, count = atomic_group(name), row["elements"]
        groups[group]["cpu"] += 2 * count  # Explicit BF16, unquantized CPU modules.
        groups[group]["gpu"] += quantized_size(count) if name in linear_weights else 2 * count
    layers = sorted((g for g in groups if ".layers." in g), key=lambda g: int(g.rsplit(".", 1)[1]))
    if not layers:
        raise PreflightError("No language decoder layers.")
    # Upstream multi-device guidance co-locates vision, projector, embeddings,
    # output and the last decoder layer. Keep the first decoder layer there too.
    required = set(groups) - set(layers) | {layers[0], layers[-1]}
    device_map = {g: (0 if g in required else "cpu") for g in groups}
    gpu = sum(groups[g]["gpu"] for g in required)
    budget = config.gpu_weight_budget_gib * GIB
    for group in layers[1:-1]:
        if gpu + groups[group]["gpu"] <= budget:
            device_map[group] = 0
            gpu += groups[group]["gpu"]
    device_map["language_model.model.rotary_emb"] = 0
    cpu = sum(row["cpu"] for g, row in groups.items() if device_map[g] == "cpu")
    gpu_needed = gpu + config.gpu_reserve_gib * GIB
    cpu_needed = cpu + config.cpu_reserve_gib * GIB
    reasons = []
    if gpu > budget:
        reasons.append("mandatory_gpu_components_exceed_weight_budget")
    if not resources["cuda_available"]:
        reasons.append("cuda_unavailable")
    if gpu_needed > resources["gpu_free_bytes"]:
        reasons.append("insufficient_free_gpu_memory")
    if cpu_needed > resources["ram_available_bytes"]:
        reasons.append("insufficient_available_system_ram")
    return {"ready_to_load": not reasons, "blockers": reasons, "device_map": device_map,
            "estimated_gpu_weight_bytes": gpu, "estimated_cpu_weight_bytes": cpu,
            "required_free_gpu_bytes_with_reserve": gpu_needed,
            "required_available_ram_bytes_with_reserve": cpu_needed,
            "cpu_offload_dtype": "bfloat16_unquantized",
            "gpu_decoder_layers": sum(device_map[g] == 0 for g in layers),
            "cpu_decoder_layers": sum(device_map[g] == "cpu" for g in layers),
            "resources": resources, "estimates_are_not_measured_peak_memory": True}


def resources():
    import torch
    memory = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()}
    available = torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info() if available else (0, 0)
    return {"cuda_available": available, "gpu_free_bytes": free, "gpu_total_bytes": total,
            "ram_available_bytes": memory["MemAvailable"], "ram_total_bytes": memory["MemTotal"]}


def parse_answer(raw):
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r"\s*(?:([AaBb])|\(([AaBb])\)|\[([AaBb])\])\.?\s*", raw)
    return next(value.upper() for value in match.groups() if value) if match else None


def tile_grid(width, height, max_tiles, size):
    ratios = sorted({(i, j) for i in range(1, max_tiles + 1) for j in range(1, max_tiles + 1)
                     if i * j <= max_tiles}, key=lambda r: (r[0] * r[1], r[0], r[1]))
    best, difference = (1, 1), math.inf
    for ratio in ratios:
        current = abs(width / height - ratio[0] / ratio[1])
        if current < difference or (current == difference and width * height >
                                   .5 * size * size * ratio[0] * ratio[1]):
            best, difference = ratio, current
    return best


def pixels(image, config):
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    image = image.convert("RGB")
    columns, rows = tile_grid(*image.size, config.max_tiles, config.image_size)
    size = config.image_size
    resized = image.resize((columns * size, rows * size), Image.Resampling.BICUBIC)
    tiles = [resized.crop((x * size, y * size, (x + 1) * size, (y + 1) * size))
             for y in range(rows) for x in range(columns)]
    if config.thumbnail and len(tiles) > 1:
        tiles.append(image.resize((size, size), Image.Resampling.BICUBIC))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[.485, .456, .406], std=[.229, .224, .225])])
    return torch.stack([transform(tile) for tile in tiles])


def validate_context(tokenizer, model, config, counts):
    question = "Image-1: <image>\nImage-2: <image>\n" + config.user_prompt
    template = model.conv_template.copy()
    template.system_message = config.system_prompt
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    text = template.get_prompt()
    for count in counts:
        text = text.replace("<image>", "<img>" + "<IMG_CONTEXT>" *
                            model.num_image_token * count + "</img>", 1)
    tokens = len(tokenizer.encode(text))
    limit = min(tokenizer.model_max_length, model.config.llm_config.max_position_embeddings)
    if tokens + config.max_new_tokens > limit:
        raise PreflightError("Image/text context exceeds the fixed limit; truncation is forbidden.")
    return question, tokens


def predict(model, tokenizer, config, image_a, image_b):
    import torch
    a, b = pixels(image_a, config), pixels(image_b, config)
    counts = [a.shape[0], b.shape[0]]
    question, tokens = validate_context(tokenizer, model, config, counts)
    device = next(model.vision_model.parameters()).device
    data = torch.cat((a, b)).to(device=device, dtype=torch.bfloat16)
    model.system_message = config.system_prompt
    start = time.monotonic()
    with torch.inference_mode():
        raw = model.chat(tokenizer, data, question,
                         {"max_new_tokens": config.max_new_tokens, "do_sample": False},
                         history=None, num_patches_list=counts)
    torch.cuda.synchronize()
    return {"raw_response": raw, "answer": parse_answer(raw), "patch_counts": counts,
            "input_tokens": tokens, "latency_seconds": time.monotonic() - start}


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             llm_int8_enable_fp32_cpu_offload=True,
                             llm_int8_skip_modules=["language_model.lm_head", "lm_head"])


def tiny_offload_probe(output, config):
    """Small random Qwen2 fixture exercises actual NF4 GPU / BF16 CPU dispatch."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    if not torch.cuda.is_available():
        raise PreflightError("The declared mixed-device probe requires CUDA.")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    fixture = Path(output) / "random-fixture"
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, eos_token_id=2, pad_token_id=0)
    cfg._attn_implementation = "eager"
    original = Qwen2ForCausalLM(cfg).to(dtype=torch.bfloat16).eval()
    original.save_pretrained(fixture, safe_serialization=True)
    del original
    mapping = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": "cpu",
               "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
    model = Qwen2ForCausalLM.from_pretrained(
        fixture, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager", quantization_config=quantization_config(), device_map=mapping).eval()
    cpu_types = {str(p.dtype) for p in model.model.layers[1].parameters()}
    gpu_types = {type(m).__name__ for m in model.model.layers[0].modules()}
    if cpu_types != {"torch.bfloat16"} or "Linear4bit" not in gpu_types:
        raise PreflightError("Synthetic mixed-device dtype/quantization does not match the planned loader.")
    inputs = torch.tensor([[1, 10, 11, 12]], device="cuda")
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        a = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=2, do_sample=False)
        b = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()
    return {"passed": bool(torch.equal(a, b)), "model": "random_tiny_two_layer_qwen2",
            "real_internvl_weights_loaded": False, "fingerprint_inputs": False,
            "cpu_parameter_dtypes": sorted(cpu_types), "gpu_linear4bit_present": True,
            "output_shape": list(a.shape), "elapsed_seconds": time.monotonic() - start,
            "peak_torch_gpu_bytes": torch.cuda.max_memory_allocated()}


def load_weights(directory, config, plan):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not plan["ready_to_load"]:
        raise PreflightError("Memory preflight is blocked; loading is refused.")
    # Check live resources again immediately before the one permitted load.
    current = resources()
    if (current["gpu_free_bytes"] < plan["required_free_gpu_bytes_with_reserve"]
            or current["ram_available_bytes"] < plan["required_available_ram_bytes_with_reserve"]):
        raise PreflightError("Resources changed after planning; no automatic smaller configuration.")
    identity = runtime_identity()
    if not identity["protected_source_clean"] or not identity["source_commit"]:
        raise PreflightError("Review and commit the protected runtime source before loading real weights.")
    model = AutoModel.from_pretrained(
        directory, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=False, quantization_config=quantization_config(),
        device_map=plan["device_map"]).eval()
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True,
                                             local_files_only=True, use_fast=False)
    for parameter in model.parameters():
        if parameter.device.type == "meta" or (parameter.device.type == "cpu" and
                parameter.is_floating_point() and parameter.dtype != torch.bfloat16):
            raise PreflightError("Actual CPU dtype/device differs from the BF16 offload plan.")
    return model, tokenizer


def cohort(bundle_path, image_root, config, *, verify_images=False, expected_per_cell=700):
    """Read fixed membership, resolving paths only inside the authorized image root."""
    bundle = json.loads(Path(bundle_path).read_text())
    if bundle.get("scope") != "numeric_group_disjoint_fixed_pairs" or bundle.get("labels") != "as_supplied":
        raise PreflightError("Use the current supplied numeric-group-holdout bundle.")
    rows, images = bundle["testing"], bundle["images"]
    ids, pairs, counts, paths = set(), set(), Counter(), {}
    image_root = Path(image_root).resolve(strict=True)
    for row in rows:
        a, b = row["image_a"], row["image_b"]
        key = tuple(sorted((a, b)))
        if (row["pair_id"] in ids or key in pairs or a == b or row["numeric_identity_disagreement"]
                or row["label"] not in {"genuine", "impostor"} or row["hand"] not in {"left", "right", "thumbs"}):
            raise PreflightError("Repeated or invalid supplied test pair.")
        hands = {"13": "right", "14": "left", "15": "thumbs"}
        suffixes = [Path(name).stem.rsplit("_", 1)[-1] for name in (a, b)]
        if (suffixes[0] != suffixes[1] or hands.get(suffixes[0]) != row["hand"]
                or (int(a.split("_")[0]) == int(b.split("_")[0])) != (row["label"] == "genuine")):
            raise PreflightError("Hand/numeric metadata differs from the preserved supplied labels.")
        ids.add(row["pair_id"]); pairs.add(key); counts[(row["hand"], row["label"])] += 1
        for name in (a, b):
            path = (image_root / images[name]["path"]).resolve(strict=True)
            if not path.is_relative_to(image_root) or not path.is_file() or path.name != name:
                raise PreflightError("Image path is missing or escapes the authorized root.")
            paths[name] = path
    expected = {(hand, label): expected_per_cell for hand in ("left", "right", "thumbs")
                for label in ("genuine", "impostor")}
    if counts != expected:
        raise PreflightError("Exact supplied left/right/thumb class counts are required.")
    train_images = {r[k] for r in bundle["training"] for k in ("image_a", "image_b")}
    test_groups = {int(PureWindowsPath(n).name.split("_")[0]) for n in paths}
    train_groups = {int(PureWindowsPath(n).name.split("_")[0]) for n in train_images}
    if set(paths) & train_images or test_groups & train_groups:
        raise PreflightError("Supplied numeric-group holdout no longer holds.")
    for name, path in paths.items():
        if verify_images:
            from PIL import Image
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != images[name]["sha256"]:
                raise PreflightError("Test image bytes changed.")
            with Image.open(io.BytesIO(data)) as image:
                image.load(); gray = image.convert("L")
                value = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
                if value != images[name]["decoded_sha256"]:
                    raise PreflightError("Decoded test image changed.")
    return rows, paths, {"pairs": len(rows), "images": len(paths), "numeric_groups": len(test_groups),
                         "bundle_sha256": digest(bundle_path), "images_verified": verify_images,
                         "config_sha256": canonical_digest(asdict(config))}


def evaluation_review(prior, review, identity_digest, bundle_digest, synthetic_digest):
    binding = {"identity": identity_digest, "bundle": bundle_digest, "synthetic_report": synthetic_digest}
    if (bundle_digest != CURRENT_BUNDLE_SHA256 or prior.get("mode") != "synthetic"
            or prior.get("status") != "passed" or prior.get("real_weights_loaded") is not True
            or prior.get("fingerprint_inference") is not False
            or prior.get("identity_sha256") != identity_digest or prior.get("synthetic_passed") is not True
            or prior.get("snapshot", {}).get("verification") != "full_sha256"
            or review.get("input_sha256") != binding or review.get("execution_authorized") is not True
            or review.get("protocol_reviewed") is not True
            or not isinstance(review.get("human_response_reference"), str)
            or not review["human_response_reference"].strip()):
        raise PreflightError("Actual protocol/execution review must bind a full-model synthetic pass and current cohort.")
    return binding


def decision_summary(rows, outcomes):
    """Failed/invalid predictions never become correct impostor decisions."""
    by_id = {row["pair_id"]: row for row in rows}
    if len(by_id) != len(rows) or len({r["pair_id"] for r in outcomes}) != len(outcomes):
        raise PreflightError("Repeated evaluation identity.")
    groups = {}
    for hand in ("pooled", "left", "right", "thumbs"):
        selected = [r for r in rows if hand == "pooled" or r["hand"] == hand]
        selected_ids = {r["pair_id"] for r in selected}
        counts, statuses = Counter(), Counter()
        for result in outcomes:
            if result["pair_id"] not in by_id:
                raise PreflightError("Unexpected evaluation outcome.")
            if result["pair_id"] not in selected_ids:
                continue
            label = by_id[result["pair_id"]]["label"]
            status, answer = result["status"], result["answer"]
            if status not in {"ok", "invalid_response", "execution_error"} or (
                    (status == "ok") != (answer in {"A", "B"})):
                raise PreflightError("Inconsistent outcome status/answer.")
            statuses[status] += 1
            if status == "ok":
                counts[("tp" if answer == "A" else "fn") if label == "genuine"
                       else ("fp" if answer == "A" else "tn")] += 1
        attempted = sum(statuses.values())
        genuine = sum(r["label"] == "genuine" for r in selected)
        impostor = len(selected) - genuine
        groups[hand] = {"expected_pairs": len(selected), "attempted": attempted,
                        "unattempted": len(selected) - attempted, "status_counts": dict(statuses),
                        "valid_confusion": {k: counts[k] for k in ("tp", "tn", "fp", "fn")},
                        "valid_coverage": statuses["ok"] / len(selected) if selected else None,
                        "accuracy_all_expected_failures_incorrect": (counts["tp"] + counts["tn"]) /
                        len(selected) if selected and attempted == len(selected) else None,
                        "tar_all_expected": counts["tp"] / genuine if genuine and attempted == len(selected) else None,
                        "far_all_expected": counts["fp"] / impostor if impostor and attempted == len(selected) else None}
    return groups


def private_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "local/precise").resolve()) or path.exists():
        raise PreflightError("Use a new output directory under ignored local/precise; no overwrite/resume.")
    path.mkdir(parents=True)
    return path


def append_event(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def execute_pairs(rows, paths, config, output, predictor):
    """One immutable attempt: journal intent first; stop on runtime error."""
    from PIL import Image
    outcomes = []
    with (Path(output) / "events.jsonl").open("x", encoding="utf-8") as events:
        for row in rows:
            pair_id = row["pair_id"]
            append_event(events, {"event": "started", "pair_id": pair_id, "utc": now()})
            try:
                with Image.open(paths[row["image_a"]]) as a, Image.open(paths[row["image_b"]]) as b:
                    result = predictor(a, b)
                result.update(pair_id=pair_id, status="ok" if result["answer"] is not None else "invalid_response")
            except Exception as error:
                result = {"pair_id": pair_id, "status": "execution_error", "answer": None,
                          "raw_response": None, "error_type": type(error).__name__, "error": str(error)}
            append_event(events, {"event": "finished", "utc": now(), **result})
            outcomes.append(result)
            if result["status"] == "execution_error":
                break
    return outcomes
