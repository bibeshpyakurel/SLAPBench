"""Verify completed selection and score final test with its unchanged cutoff.

No split creation, checkpoint selection or test-threshold optimization exists here.
Identifying outcomes and the exclusive test claim belong in private run storage.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .data import InputError, load_protocol, read_table, validate_protocol
from .encoders import CONFIG_SHA256, MODEL_ID, REVISION, WEIGHTS_SHA256, cosine_scores, encode
from .evaluation import Score, evaluate_development, summarize
from .training import Config, digest, selection_key, write_json


def artifact(root, relative, pattern):
    if not isinstance(relative, str) or not re.fullmatch(pattern, relative):
        raise InputError("Invalid training artifact reference.")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise InputError("Training artifact escapes its private run.")
    return path


def inspect_selection(run_root, selection_path, config, hashes):
    """Metadata/digest inspection before importing torch or opening test images."""
    root = Path(run_root).resolve(strict=True)
    selection = Path(selection_path).resolve(strict=True)
    if not selection.is_relative_to(root):
        raise InputError("Selection must belong to the supplied training run.")
    selection = artifact(root, str(selection.relative_to(root)), r"attempt-\d+/result\.json")
    attempts = [p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"attempt-\d+", p.name)]
    latest = max(attempts, key=lambda p: int(p.name.split("-")[1]))
    if selection.parent != latest or (latest / "failure.json").exists():
        raise InputError("Use the latest successful completed training attempt.")
    record_path = artifact(root, "run_record.json", r"run_record\.json")
    record = json.loads(record_path.read_text())
    result = json.loads(selection.read_text())
    if not isinstance(record, dict) or not isinstance(result, dict):
        raise InputError("Training records must be JSON objects.")
    if (result.get("training_complete") is not True or result.get("selection_frozen") is not True
            or result.get("test_scores_accessed") is not False
            or type(result.get("epochs_completed")) is not int
            or result.get("epochs_completed") != config.epochs
            or record.get("config") != asdict(config) or record.get("schema_version") != 1):
        raise InputError("Training and selection must be complete and frozen under the reviewed configuration.")
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict) or not isinstance(provenance.get("input_sha256"), dict):
        raise InputError("Missing training provenance.")
    expected = {"model": MODEL_ID, "model_revision": REVISION,
                "weights_sha256": WEIGHTS_SHA256, "model_config_sha256": CONFIG_SHA256}
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise InputError("Selected training model provenance does not match the pinned encoder.")
    if any(provenance.get("input_sha256", {}).get(k) != hashes[k] for k in ("images", "pairs", "config")):
        raise InputError("Training and test must use the same reviewed protocol/configuration bytes.")
    best = result["selected"]
    if not isinstance(best, dict):
        raise InputError("Selected checkpoint metadata must be an object.")
    epoch = best.get("epoch")
    if type(epoch) is not int or not 1 <= epoch <= config.epochs:
        raise InputError("Invalid selected epoch.")
    checkpoint = artifact(root, best["checkpoint"], r"attempt-\d+/epoch-\d+\.pt")
    last = artifact(root, result["last_checkpoint"], r"attempt-\d+/epoch-\d+\.pt")
    if checkpoint.name != f"epoch-{epoch:04d}.pt" or last.name != f"epoch-{config.epochs:04d}.pt":
        raise InputError("Checkpoint paths disagree with selected/completed epochs.")
    development = artifact(root, str(checkpoint.with_name(
        f"development-{epoch:04d}.json").relative_to(root)), r"attempt-\d+/development-\d+\.json")
    files = {"selection": selection, "run_record": record_path, "checkpoint": checkpoint,
             "last_checkpoint": last, "development": development}
    values = {name: digest(path) for name, path in files.items()}
    for name in ("checkpoint", "last_checkpoint"):
        if files[name].with_suffix(".sha256").read_text().strip() != values[name]:
            raise InputError("Training checkpoint digest mismatch.")
    if values["checkpoint"] != result["selected_checkpoint_sha256"]:
        raise InputError("Selected checkpoint differs from the frozen selection.")
    return {"root": root, "record": record, "result": result, "files": files, "hashes": values}


def reviewed_test_inputs(images_path, pairs_path, paths_path, config_path, review_path,
                         source_root, run_root, selection_path):
    sources = {"images": Path(images_path), "pairs": Path(pairs_path),
               "paths": Path(paths_path), "config": Path(config_path)}
    hashes = {k: digest(p) for k, p in sources.items()}
    config = Config(**json.loads(sources["config"].read_text()))
    config.validate()
    info = inspect_selection(run_root, selection_path, config, hashes)
    hashes.update(info["hashes"])
    review = json.loads(Path(review_path).read_text())
    flags = ("protocol_reviewed", "split_reviewed", "capture_exposure_reviewed",
             "execution_authorized", "selection_reviewed", "final_test_authorized")
    if (not isinstance(review, dict) or any(review.get(flag) is not True for flag in flags)
            or review.get("input_sha256") != hashes
            or not isinstance(review.get("review_record"), str) or not review["review_record"].strip()):
        raise InputError("Final-test review must bind the exact protocol, paths and frozen training artifacts.")
    images, pairs = load_protocol(images_path, pairs_path)
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Final-test metadata has unresolved capture/exposure/partition blockers.")
    by_id = {i.image_id: i for i in images}
    needed = {key for p in pairs if by_id[p.image_a].partition == "test" for key in (p.image_a, p.image_b)}
    root = Path(source_root).resolve(strict=True)
    paths = {}
    for row in read_table(Path(paths_path), ("image_id", "path")):
        key = row["image_id"]
        if key not in needed or key in paths or not row["path"]:
            raise InputError("Final-test paths must cover test endpoints only, exactly once.")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise InputError("Test image path escapes the reviewed source root.")
        paths[key] = path
    if set(paths) != needed:
        raise InputError("Final-test path mapping is incomplete.")
    all_sources = {**sources, **info["files"], "review": Path(review_path)}
    hashes["review"] = digest(review_path)

    def integrity():
        if {k: digest(p) for k, p in all_sources.items()} != hashes:
            raise InputError("Frozen evaluation inputs or training artifacts changed.")
        for key in ("checkpoint", "last_checkpoint"):
            if info["files"][key].with_suffix(".sha256").read_text().strip() != hashes[key]:
                raise InputError("Checkpoint digest sidecar changed.")

    integrity()
    return images, pairs, paths, config, info, hashes, integrity


def restore_selection(model, info, images, pairs, config):
    """Check checkpoint history and replay selected DEVELOPMENT scores before test."""
    import torch

    if {k: digest(p) for k, p in info["files"].items()} != info["hashes"]:
        raise InputError("Frozen training artifact bytes changed before checkpoint restoration.")
    saved = {}
    for key in ("checkpoint", "last_checkpoint"):
        saved[key] = torch.load(info["files"][key], map_location="cpu", weights_only=True)
        if saved[key]["record"] != info["record"]:
            raise InputError("Checkpoint configuration/provenance mismatch.")
    selected, final = saved["checkpoint"], saved["last_checkpoint"]
    best = info["result"]["selected"]
    epoch = best["epoch"]
    history = final["history"]
    if (type(final["epochs_completed"]) is not int or final["epochs_completed"] != config.epochs
            or len(history) != config.epochs
            or [x["epoch"] for x in history] != list(range(1, config.epochs + 1))
            or final["sampler_position"] != config.epochs * config.microbatches
            or final["scheduler"]["last_epoch"] != config.steps
            or selected["epochs_completed"] != epoch or selected["history"] != history[:epoch]
            or final["best"] != best or selected["best"] != best):
        raise InputError("Selected/completed checkpoints do not establish the frozen training history.")
    if any(row["development"].get("selection_source") != "development_only" for row in history):
        raise InputError("Checkpoint selection must use development only.")
    winner = max(history, key=lambda row: selection_key(row["development"], row["epoch"]))
    key = list(selection_key(winner["development"], winner["epoch"]))
    if winner["epoch"] != epoch or best["key"] != key:
        raise InputError("Selected checkpoint disagrees with the recorded development selection rule.")
    data = json.loads(info["files"]["development"].read_text())
    scores = [Score(**row) for row in data["scores"]]
    report = evaluate_development(images, pairs, scores, config.target_far)
    if (report != data["metrics"] or report != history[epoch - 1]["development"]
            or report["threshold"] != best["threshold"]):
        raise InputError("Frozen cutoff differs from the selected checkpoint's development scores.")
    cutoff = best["threshold"]["cutoff"]
    if not math.isfinite(cutoff) or best["threshold"]["comparison"] != ">":
        raise InputError("Expected a finite development-selected strict-greater-than cutoff.")
    model.load_state_dict(selected["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return report


class ImageFailure(Exception):
    def __init__(self, status):
        if status not in {"decode_error", "missing_input"}:
            raise InputError("Invalid test image failure category.")
        self.status = status


def score_test(model, images, pairs, pixel_loader, device, event=lambda row: None):
    """Read each needed TEST image once; preserve every attempted comparison."""
    import torch

    by_id = {i.image_id: i for i in images}
    selected = [p for p in pairs if by_id[p.image_a].partition == "test"]
    cache = {}
    model.eval()
    with torch.inference_mode():
        for key in sorted({key for p in selected for key in (p.image_a, p.image_b)}):
            try:
                pixels = pixel_loader(by_id[key])
            except ImageFailure as error:
                cache[key] = (None, error.status)
                event({"kind": "image", "image_id": key, "status": error.status})
                continue
            except Exception:
                event({"kind": "aborted", "image_id": key, "stage": "image_load"})
                raise
            try:
                cache[key] = (encode(model, pixels.unsqueeze(0).to(device)).cpu(), "ok")
            except InputError:
                cache[key] = (None, "encoder_error")
            except Exception:
                event({"kind": "aborted", "image_id": key, "stage": "encoder"})
                raise
            event({"kind": "image", "image_id": key, "status": cache[key][1]})
    scores = []
    for pair in selected:
        a, b = cache[pair.image_a], cache[pair.image_b]
        failures = {a[1], b[1]} - {"ok"}
        if failures:
            status = next(s for s in ("missing_input", "decode_error", "encoder_error") if s in failures)
            scores.append(Score(pair.pair_id, status, None))
        else:
            scores.append(Score(pair.pair_id, "ok", float(cosine_scores(a[0], b[0])[0])))
        event({"kind": "comparison", **asdict(scores[-1])})
    return selected, scores


def final_test(model, images, pairs, pixel_loader, config, info, output, hashes, *, integrity=lambda: None):
    """One private claim per adapted run; failed attempts remain consumed/visible."""
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Unreviewed protocol cannot enter final test.")
    integrity()
    development = restore_selection(model, info, images, pairs, config)
    integrity()
    output = Path(output).resolve()
    claim = info["root"] / "final_test_adapted_claim.json"
    if output.exists() or output.is_relative_to(info["root"]) or claim.exists():
        raise InputError("Use a new separate output; a claimed adapted final test cannot be repeated automatically.")
    output.mkdir(parents=True)
    # Exclusive claim is made before opening any test pixels. A failed claimed
    # run cannot be silently repeated using a different output directory.
    write_json(claim, {"output": str(output), "input_sha256": hashes,
                       "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "selected_checkpoint_sha256": info["hashes"]["checkpoint"]})
    try:
        write_json(output / "frozen_selection.json", {"selected": info["result"]["selected"],
            "input_sha256": hashes, "checkpoint_verified": True,
            "selected_development_replayed": True})
        with (output / "events.jsonl").open("x") as stream:
            def event(row):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()

            selected, scores = score_test(model, images, pairs, pixel_loader, config.device, event)
        integrity()
        write_json(output / "scores.json", [asdict(row) for row in scores])
        cutoff = development["threshold"]["cutoff"]
        by_score = {row.pair_id: row for row in scores}
        by_id = {row.image_id: row for row in images}
        rows = [(p.label, by_score[p.pair_id]) for p in selected]
        report = {"schema_version": 1, "scope": "Adapted matcher final test with frozen development cutoff",
                  "threshold": development["threshold"], "selected_epoch": info["result"]["selected"]["epoch"],
                  "test": summarize(rows, cutoff),
                  "by_hand_same_pooled_cutoff": {hand: summarize(
                      [(p.label, by_score[p.pair_id]) for p in selected
                       if by_id[p.image_a].hand == hand], cutoff) for hand in ("left", "right")},
                  "selection_record_verified": True, "test_used_for_selection": False,
                  "limits": ["Private review assertions still require real underlying acquisition/exposure evidence.",
                             "Failures count as rejects with coverage reported; they do not establish biometric impostor rejection.",
                             "Test ROC/DET/EER are descriptive; no test-driven threshold or checkpoint selection.",
                             "No dependence-aware confidence interval or population low-FAR bound is supplied."]}
        integrity()
        write_json(output / "report.json", report)
        return report
    except Exception:
        write_json(output / "failure.json", {"status": "failed", "claim_retained": True,
                    "automatic_retry_allowed": False})
        raise
