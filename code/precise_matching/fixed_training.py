"""Exact supplied-pair adaptation with a predeclared final checkpoint.

No testing pairs, labels, pixels or scores are arguments to this training loop.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re

from .data import InputError
from .encoders import cosine_scores, encode
from .fixed_pairs import canonical_digest, validate_pairs
from .training import digest, save_checkpoint, write_json


def explicit_pair_loss(a, b, labels, margin):
    """Mean per supplied pair, retaining single-class and incomplete batches."""
    import torch

    if len(labels) != a.shape[0] or not labels or any(label not in {"genuine", "impostor"} for label in labels):
        raise InputError("Explicit targets must align with the supplied pairs.")
    scores = cosine_scores(a, b)
    positive = torch.tensor([label == "genuine" for label in labels], device=scores.device)
    return torch.where(positive, 1 - scores, (scores - margin).clamp_min(0)).mean()


def shuffled_batches(pairs, batch_pairs, seed, epoch):
    """Every supplied pair occurs once; no extra comparisons or dropped tails."""
    order = list(range(len(pairs)))
    random.Random(seed + epoch).shuffle(order)
    return [[pairs[i] for i in order[start:start + batch_pairs]]
            for start in range(0, len(order), batch_pairs)]


def checked_checkpoint(path, output, record):
    import torch

    path, output = Path(path).resolve(), Path(output).resolve()
    if (not path.is_relative_to(output) or not path.is_file()
            or path.with_suffix(".sha256").read_text().strip() != digest(path)):
        raise InputError("Checkpoint must be a digest-verified artifact of this run.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["record"] != record:
        raise InputError("Checkpoint configuration/provenance mismatch.")
    return saved


def train_fixed(model, pairs, pixel_loader, config, output, provenance, *,
                integrity=lambda: None, resume=None, stop_after_epochs=None, event=lambda row: None):
    """Train with explicit-pair supervision; synthetic tests inject a tiny model."""
    import numpy as np
    import torch

    validate_pairs(pairs)
    config.validate(len(pairs))
    integrity()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise InputError("CUDA unavailable; no automatic fallback.")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise InputError("BF16 unavailable; no automatic fallback.")
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    model.to(config.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise InputError("Adaptation requires trainable parameters.")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim > 1], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(.9, .999), eps=1e-8)
    steps = config.steps(len(pairs))

    def multiplier(step):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = min(1, (step - config.warmup_steps) / max(1, steps - config.warmup_steps - 1))
        ratio = config.minimum_learning_rate / config.learning_rate
        return ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = Path(output).resolve()
    record = json.loads(json.dumps({"schema_version": 1, "route": "explicit_fixed_pairs",
        "config": asdict(config), "training_pairs_sha256": canonical_digest([asdict(p) for p in pairs]),
        "provenance": provenance}, allow_nan=False))
    history, start = [], 0
    updates_per_epoch = steps // config.epochs
    if resume is None:
        if output.exists():
            raise InputError("A fresh run requires a new private directory.")
        output.mkdir(parents=True)
        write_json(output / "run_record.json", record)
    else:
        if json.loads((output / "run_record.json").read_text()) != record or (output / "selection.json").exists():
            raise InputError("Resume requires unchanged provenance and an unfinished run.")
        saved = checked_checkpoint(resume, output, record)
        start, history = saved["epochs_completed"], saved["history"]
        if (not 0 < start < config.epochs or len(history) != start
                or saved["scheduler"]["last_epoch"] != start * updates_per_epoch
                or [h["epoch"] for h in history] != list(range(1, start + 1))
                or any(h["pairs_seen"] != len(pairs) for h in history)):
            raise InputError("Resume requires a verified unfinished epoch boundary.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        random.setstate(saved["python_rng"])
        state = saved["numpy_rng"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(saved["torch_rng"].cpu())
        if config.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    attempts = [int(p.name.split("-")[1]) for p in output.iterdir()
                if p.is_dir() and re.fullmatch(r"attempt-\d+", p.name)]
    attempt = output / f"attempt-{max(attempts, default=0) + 1:04d}"
    attempt.mkdir()
    write_json(attempt / "attempt.json", {"start_epoch": start,
        "resume_sha256": digest(resume) if resume else None,
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    completed = start
    try:
        limit = config.epochs if stop_after_epochs is None else min(config.epochs, start + stop_after_epochs)
        if limit <= start:
            raise InputError("An attempt must finish at least one epoch.")
        for epoch in range(start, limit):
            integrity()
            model.train()
            batches = shuffled_batches(pairs, config.batch_pairs, config.seed, epoch)
            loss_sum, seen = 0.0, 0
            for window_start in range(0, len(batches), config.accumulation):
                window = batches[window_start:window_start + config.accumulation]
                window_pairs = sum(len(batch) for batch in window)
                optimizer.zero_grad(set_to_none=True)
                for batch in window:
                    # Only explicitly aligned endpoints enter the loss. Reused endpoints
                    # retain their occurrence weight; no identity-based cross-pairs exist.
                    keys = [key for p in batch for key in (p.image_a, p.image_b)]
                    pixels = torch.stack([pixel_loader(key) for key in keys]).to(config.device)
                    with torch.autocast(config.device, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
                        vectors = encode(model, pixels)
                    loss = explicit_pair_loss(vectors[0::2], vectors[1::2], [p.label for p in batch], config.margin)
                    if not torch.isfinite(loss):
                        raise InputError("Nonfinite explicit-pair loss.")
                    (loss * (len(batch) / window_pairs)).backward()
                    loss_sum += float(loss.detach()) * len(batch)
                    seen += len(batch)
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise InputError("Missing or nonfinite adaptation gradients.")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                event({"epoch": epoch + 1, "pairs_seen": seen, "pairs_total": len(pairs),
                       "updates_completed": scheduler.last_epoch, "updates_total": steps})
            if seen != len(pairs):
                raise InputError("A supplied pair was omitted or repeated.")
            integrity()
            history.append({"epoch": epoch + 1, "pairs_seen": seen,
                "mean_pair_loss": loss_sum / seen, "learning_rate": optimizer.param_groups[0]["lr"]})
            numpy_rng = np.random.get_state()
            checkpoint = attempt / f"epoch-{epoch + 1:04d}.pt"
            save_checkpoint(checkpoint, {"record": record, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epochs_completed": epoch + 1, "history": history,
                "python_rng": random.getstate(), "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else []})
            completed = epoch + 1
            write_json(attempt / f"epoch-{completed:04d}.json", history[-1])
        integrity()
        result = {"epochs_completed": completed, "training_complete": completed == config.epochs,
            "checkpoint_rule": config.checkpoint_rule, "test_scores_accessed": False,
            "checkpoint": str(checkpoint.relative_to(output)), "checkpoint_sha256": digest(checkpoint),
            "history": history, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(attempt / "result.json", result)
        if completed == config.epochs:
            # A single final selection seals the run; no best-epoch comparison.
            write_json(output / "selection.json", result)
        return result
    except BaseException as exc:
        write_json(attempt / "failure.json", {"status": "failed", "exception_type": type(exc).__name__,
            "last_checkpointed_epoch": completed, "test_scores_accessed": False})
        raise
