"""Reviewed-input training with development-only selection and immutable attempts.

No split/pair generation or final-test image/score loader exists in this module.
Model execution requires the caller's separate review and execution assertions.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile

from .data import InputError, load_protocol, read_table, validate_protocol
from .encoders import cosine_scores, encode
from .evaluation import Score, evaluate_development, target_fraction
from .losses import verification_loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class Config:
    seed: int = 20261025
    epochs: int = 30
    microbatches: int = 200
    subjects_per_batch: int = 8
    accumulation: int = 4
    warmup_steps: int = 100
    learning_rate: float = 1e-5
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    margin: float = 0.2
    target_far: str = "0.001"
    device: str = "cuda"
    precision: str = "bf16"

    def validate(self):
        values = (self.seed, self.epochs, self.microbatches, self.subjects_per_batch,
                  self.accumulation, self.warmup_steps)
        if any(type(v) is not int for v in values):
            raise InputError("Integer training configuration fields required.")
        if (not 0 <= self.seed < 2**63 or min(self.epochs, self.microbatches, self.accumulation) < 1
                or self.subjects_per_batch < 2 or self.microbatches % self.accumulation
                or self.microbatches % 2 or not 0 <= self.warmup_steps < self.steps):
            raise InputError("Invalid training schedule or batch configuration.")
        if (not all(type(v) in {int, float} and math.isfinite(v) for v in (self.learning_rate, self.minimum_learning_rate,
                                              self.weight_decay, self.margin))
                or not 0 < self.minimum_learning_rate <= self.learning_rate
                or self.weight_decay < 0 or not -1 <= self.margin <= 1):
            raise InputError("Invalid optimizer or verification-loss settings.")
        if self.device not in {"cpu", "cuda"} or self.precision not in {"fp32", "bf16"}:
            raise InputError("Explicit device/precision required.")
        if self.device == "cpu" and self.precision != "fp32":
            raise InputError("CPU execution requires an explicitly reviewed FP32 configuration.")
        target_fraction(self.target_far)

    @property
    def steps(self):
        return self.epochs * self.microbatches // self.accumulation


def reviewed_inputs(images_path, pairs_path, paths_path, config_path, review_path, source_root):
    """Require review assertions bound to exact input bytes before any model load."""
    sources = {"images": images_path, "pairs": pairs_path, "paths": paths_path, "config": config_path}
    hashes = {name: digest(path) for name, path in sources.items()}
    review_bytes = Path(review_path).read_bytes()
    review = json.loads(review_bytes)
    flags = ("protocol_reviewed", "split_reviewed", "capture_exposure_reviewed", "execution_authorized")
    if (not isinstance(review, dict) or any(review.get(flag) is not True for flag in flags)
            or review.get("input_sha256") != hashes or not isinstance(review.get("review_record"), str)
            or not review["review_record"].strip()):
        raise InputError("Explicit review assertions must bind every current input digest.")
    config = Config(**json.loads(Path(config_path).read_text()))
    config.validate()
    images, pairs = load_protocol(images_path, pairs_path)
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Capture/exposure/partition metadata has unresolved blockers.")
    by_id = {row.image_id: row for row in images}
    dev_ids = {key for p in pairs if by_id[p.image_a].partition == "development"
               for key in (p.image_a, p.image_b)}
    allowed = {i.image_id for i in images if i.partition == "train"} | dev_ids
    paths = {}
    root = Path(source_root).resolve(strict=True)
    for row in read_table(Path(paths_path), ("image_id", "path")):
        key = row["image_id"]
        if key not in allowed or key in paths or not row["path"]:
            raise InputError("Path mapping must contain training/development inputs only, exactly once.")
        # Do not open image files here. Final-test paths are never accepted.
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise InputError("Image path escapes the reviewed source root.")
        paths[key] = path
    if set(paths) != allowed:
        raise InputError("Reviewed training/development image paths are incomplete.")
    hashes["review"] = hashlib.sha256(review_bytes).hexdigest()

    def integrity():
        current = {name: digest(path) for name, path in sources.items()}
        current["review"] = digest(review_path)
        if current != hashes:
            raise InputError("Reviewed inputs changed during execution.")

    integrity()
    return images, pairs, paths, config, hashes, integrity


class Sampler:
    def __init__(self, images, subjects_per_batch, seed):
        groups = defaultdict(lambda: defaultdict(list))
        seen_captures = set()
        for row in sorted(images, key=lambda r: r.image_id):
            if row.partition != "train":
                continue
            if row.capture_id in seen_captures:
                raise InputError("Training requires one explicitly reviewed canonical image per capture.")
            seen_captures.add(row.capture_id)
            groups[row.hand][row.subject_id].append(row)
        self.groups = {hand: {s: rows for s, rows in groups[hand].items() if len(rows) >= 2}
                       for hand in ("left", "right")}
        if (any(len(rows) < 2 for group in groups.values() for rows in group.values())
                or set(groups["left"]) != set(groups["right"])):
            raise InputError("Review training eligibility for both hands; subjects are never silently omitted.")
        if any(len(group) < subjects_per_batch for group in self.groups.values()):
            raise InputError("Both hands require enough subjects with independent training captures.")
        self.count = subjects_per_batch
        self.rng = random.Random(seed)
        self.position = 0

    def next(self):
        hand = ("left", "right")[self.position % 2]
        group = self.groups[hand]
        subjects = self.rng.sample(sorted(group), self.count)
        rows = [row for subject in subjects for row in self.rng.sample(group[subject], 2)]
        self.position += 1
        return rows


def development_scores(model, images, pairs, pixel_loader, device):
    import torch

    by_id = {r.image_id: r for r in images}
    selected = [p for p in pairs if by_id[p.image_a].partition == "development"]
    needed = sorted({key for p in selected for key in (p.image_a, p.image_b)})
    embeddings = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for key in needed:
                embeddings[key] = encode(model, pixel_loader(by_id[key]).unsqueeze(0).to(device)).cpu()
            return [Score(p.pair_id, "ok", float(cosine_scores(
                embeddings[p.image_a], embeddings[p.image_b])[0])) for p in selected]
    finally:
        model.train(was_training)


def selection_key(report, epoch):
    dev = report["development"]
    return (dev["all_attempts_failure_as_reject"]["tar"],
            -dev["descriptive_valid_score_discrimination"]["eer_linear_interpolation"], -epoch)


def save_checkpoint(path, payload):
    """Install a fully written checkpoint exclusively; never replace an artifact."""
    import torch

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)  # fails atomically if the destination exists
        finally:
            temporary.unlink(missing_ok=True)
    with path.with_suffix(".sha256").open("x") as stream:
        stream.write(digest(path) + "\n")


def train(model, images, pairs, pixel_loader, config, output, provenance, *,
          integrity=lambda: None, resume=None, stop_after_epochs=None):
    """Run the reviewed loop; tests inject synthetic inputs and a tiny encoder."""
    import numpy as np
    import torch

    config.validate()
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Unreviewed protocol metadata cannot enter training.")
    integrity()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise InputError("CUDA unavailable; automatic CPU fallback is prohibited.")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise InputError("BF16 unavailable; revise the reviewed precision configuration.")
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    model.to(config.device)
    sampler = Sampler(images, config.subjects_per_batch, config.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise InputError("Adapted training requires trainable parameters.")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim > 1], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    def multiplier(step):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = min(1, (step - config.warmup_steps) /
                       max(1, config.steps - config.warmup_steps - 1))
        ratio = config.minimum_learning_rate / config.learning_rate
        return ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = Path(output).resolve()
    # Canonical JSON primitives also exclude unsafe pickle-only metadata classes
    # such as torch's str-subclass TorchVersion from weights-only checkpoints.
    record = json.loads(json.dumps(
        {"schema_version": 1, "config": asdict(config), "provenance": provenance}, allow_nan=False))
    history, best, start = [], None, 0
    if resume is None:
        if output.exists():
            raise InputError("New training outputs require a new directory.")
        output.mkdir(parents=True)
        write_json(output / "run_record.json", record)
    else:
        resume = Path(resume).resolve()
        if not resume.is_relative_to(output) or not resume.is_file():
            raise InputError("Resume requires a checkpoint from this run.")
        if json.loads((output / "run_record.json").read_text()) != record:
            raise InputError("Resume configuration or provenance changed.")
        if resume.with_suffix(".sha256").read_text().strip() != digest(resume):
            raise InputError("Resume checkpoint digest mismatch.")
        saved = torch.load(resume, map_location=config.device, weights_only=True)
        if saved["record"] != record:
            raise InputError("Resume checkpoint provenance mismatch.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, history, best = saved["epochs_completed"], saved["history"], saved["best"]
        if (not 0 < start < config.epochs or saved["sampler_position"] != start * config.microbatches
                or len(history) != start or scheduler.last_epoch != start * config.microbatches // config.accumulation):
            raise InputError("Resume position is not a valid unfinished epoch boundary.")
        sampler.rng.setstate(saved["sampler_rng"])
        sampler.position = saved["sampler_position"]
        random.setstate(saved["python_rng"])
        state = saved["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(saved["torch_rng"].cpu())
        if config.device == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    attempts = [int(p.name.split("-")[1]) for p in output.iterdir()
                if p.is_dir() and re.fullmatch(r"attempt-\d+", p.name)]
    attempt = output / f"attempt-{max(attempts, default=0) + 1:04d}"
    attempt.mkdir()
    write_json(attempt / "attempt.json", {"resume_sha256": digest(resume) if resume else None,
                                        "start_epoch": start,
                                        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    last_saved_epoch = start
    try:
        limit = config.epochs if stop_after_epochs is None else min(config.epochs, start + stop_after_epochs)
        if limit <= start:
            raise InputError("An attempt must complete at least one epoch.")
        for epoch in range(start, limit):
            integrity()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro in range(config.microbatches):
                rows = sampler.next()
                pixels = torch.stack([pixel_loader(row) for row in rows]).to(config.device)
                with torch.autocast(config.device, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                    embeddings = encode(model, pixels)
                loss, _ = verification_loss(embeddings, [r.subject_id for r in rows],
                                            [r.capture_id for r in rows], [r.hand for r in rows], config.margin)
                if not torch.isfinite(loss):
                    raise InputError("Nonfinite training loss.")
                loss_sum += float(loss.detach())
                (loss / config.accumulation).backward()
                if (micro + 1) % config.accumulation == 0:
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                        raise InputError("Missing or nonfinite training gradients.")
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            scores = development_scores(model, images, pairs, pixel_loader, config.device)
            report = evaluate_development(images, pairs, scores, config.target_far)
            integrity()
            write_json(attempt / f"development-{epoch + 1:04d}.json",
                       {"metrics": report, "scores": [asdict(s) for s in scores]})
            checkpoint = attempt / f"epoch-{epoch + 1:04d}.pt"
            key = selection_key(report, epoch + 1)
            if best is None or key > tuple(best["key"]):
                best = {"key": list(key), "epoch": epoch + 1,
                        "checkpoint": str(checkpoint.relative_to(output)), "threshold": report["threshold"]}
            history.append({"epoch": epoch + 1, "mean_training_loss": loss_sum / config.microbatches,
                            "development": report, "learning_rate": optimizer.param_groups[0]["lr"]})
            numpy_rng = np.random.get_state()
            save_checkpoint(checkpoint, {"record": record, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epochs_completed": epoch + 1, "history": history, "best": best,
                "sampler_rng": sampler.rng.getstate(), "sampler_position": sampler.position,
                "python_rng": random.getstate(), "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else []})
            last_saved_epoch = epoch + 1
        integrity()
        selected = (output / best["checkpoint"]).resolve()
        if not selected.is_relative_to(output):
            raise InputError("Selected checkpoint escapes this private run.")
        if digest(selected) != selected.with_suffix(".sha256").read_text().strip():
            raise InputError("Selected checkpoint digest mismatch.")
        result = {"epochs_completed": limit, "training_complete": limit == config.epochs,
                  "selection_frozen": limit == config.epochs,
                  "last_checkpoint": str(checkpoint.relative_to(output)),
                  "selected": best, "selected_checkpoint_sha256": digest(selected),
                  "test_scores_accessed": False,
                  "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(attempt / "result.json", result)
        return result
    except Exception:
        write_json(attempt / "failure.json", {"status": "failed", "last_checkpointed_epoch": last_saved_epoch,
                                             "test_scores_accessed": False})
        raise
