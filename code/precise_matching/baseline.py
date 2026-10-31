"""Two-stage frozen control: development selection, then separately reviewed test.

No training, allocation, model search or test-driven cutoff fitting is performed.
Review assertions require underlying evidence; they are never generated here.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .data import InputError, load_protocol, read_table, validate_protocol
from .encoders import CONFIG_SHA256, MODEL_ID, REVISION, WEIGHTS_SHA256
from .evaluation import Score, evaluate_development, summarize, write_scores
from .testing import score_partition
from .training import Config, digest, write_json


MODEL = {"model": MODEL_ID, "model_revision": REVISION,
         "weights_sha256": WEIGHTS_SHA256, "model_config_sha256": CONFIG_SHA256}
ARTIFACTS = {"baseline_record": "run_record.json", "baseline_selection": "selection.json",
             "baseline_development": "development_scores.json", "baseline_report": "report.json"}
COMPATIBILITY = (*MODEL, "source_sha256", "packages", "python", "environment_lock_sha256", "source_root")


def inspect_baseline(run_root, images, pairs, config, hashes):
    """Verify immutable baseline records and replay DEVELOPMENT scores only."""
    root = Path(run_root).resolve(strict=True)
    files = {key: (root / name).resolve(strict=True) for key, name in ARTIFACTS.items()}
    if (any(not p.is_relative_to(root) or not p.is_file() for p in files.values())
            or (root / "failure.json").exists()):
        raise InputError("Use a completed private baseline with contained immutable artifacts.")
    artifact_hashes = {k: digest(p) for k, p in files.items()}
    record = json.loads(files["baseline_record"].read_text())
    selection = json.loads(files["baseline_selection"].read_text())
    completed = json.loads(files["baseline_report"].read_text())
    if not isinstance(record, dict) or not isinstance(selection, dict):
        raise InputError("Baseline records must be JSON objects.")
    if (record.get("schema_version") != 1 or record.get("arm") != "frozen_dinov2"
            or record.get("phase") != "development" or record.get("inference_precision") != "fp32"
            or record.get("optimizer_created") is not False
            or record.get("config") != asdict(config)
            or not isinstance(record.get("provenance"), dict)
            or any(record["provenance"].get(k) != v for k, v in MODEL.items())
            or not isinstance(record.get("input_sha256"), dict)
            or any(record["input_sha256"].get(k) != hashes[k] for k in ("images", "pairs", "config"))
            or selection.get("development_complete") is not True
            or selection.get("selection_frozen") is not True
            or selection.get("test_scores_accessed") is not False
            or selection.get("record_sha256") != artifact_hashes["baseline_record"]
            or selection.get("development_scores_sha256") != artifact_hashes["baseline_development"]):
        raise InputError("Frozen baseline selection or protocol/configuration binding is invalid.")
    scores = [Score(**row) for row in json.loads(files["baseline_development"].read_text())]
    report = evaluate_development(images, pairs, scores, config.target_far)
    if selection.get("development") != report or selection.get("threshold") != report["threshold"]:
        raise InputError("Frozen baseline cutoff differs from its original development scores.")
    if completed != {"schema_version": 1, "scope": "Frozen encoder development only; no test scoring",
                     "development": report, "test_scores_accessed": False}:
        raise InputError("Baseline development must have a completed matching report.")
    return {"root": root, "files": files, "hashes": artifact_hashes,
            "record": record, "development": report, "scores": scores}


def reviewed_baseline_inputs(images_path, pairs_path, paths_path, config_path, review_path,
                             source_root, phase, baseline_run=None):
    """Digest-bound phase review and path-population checks before model loading."""
    if phase not in {"development", "test"} or (phase == "test") != (baseline_run is not None):
        raise InputError("Baseline testing requires a prior development run; development accepts no test selection.")
    sources = {"images": Path(images_path), "pairs": Path(pairs_path),
               "paths": Path(paths_path), "config": Path(config_path)}
    hashes = {k: digest(p) for k, p in sources.items()}
    config = Config(**json.loads(sources["config"].read_text()))
    config.validate()
    images, pairs = load_protocol(images_path, pairs_path)
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Baseline metadata has unresolved capture/exposure/partition blockers.")
    info = None
    if phase == "test":
        if (Path(baseline_run) / "final_test_frozen_claim.json").exists():
            raise InputError("The frozen final test is already claimed; automatic repeats are prohibited.")
        info = inspect_baseline(baseline_run, images, pairs, config, hashes)
        sources.update(info["files"])
        hashes.update(info["hashes"])
    review = json.loads(Path(review_path).read_text())
    flags = ["protocol_reviewed", "split_reviewed", "capture_exposure_reviewed",
             "execution_authorized", "frozen_control_reviewed"]
    if phase == "test":
        flags += ["selection_reviewed", "final_test_authorized"]
    if (not isinstance(review, dict) or review.get("phase") != phase
            or any(review.get(k) is not True for k in flags) or review.get("input_sha256") != hashes
            or not isinstance(review.get("review_record"), str) or not review["review_record"].strip()):
        raise InputError("Baseline phase review must bind all current inputs and any frozen development artifacts.")
    root = Path(source_root).resolve(strict=True)
    by_id = {i.image_id: i for i in images}
    needed = {key for p in pairs if by_id[p.image_a].partition == phase for key in (p.image_a, p.image_b)}
    paths = {}
    for row in read_table(Path(paths_path), ("image_id", "path")):
        key = row["image_id"]
        if key not in needed or key in paths or not row["path"]:
            raise InputError("Baseline paths must cover only the current phase's comparison endpoints, exactly once.")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise InputError("Baseline image path escapes the reviewed source root.")
        paths[key] = path
    if set(paths) != needed:
        raise InputError("Baseline phase paths are incomplete.")
    sources["review"] = Path(review_path)
    hashes["review"] = digest(review_path)

    def integrity():
        if {k: digest(p) for k, p in sources.items()} != hashes:
            raise InputError("Reviewed baseline inputs or frozen development artifacts changed.")

    integrity()
    return images, pairs, paths, config, info, hashes, integrity


def run_baseline(model, images, pairs, pixel_loader, config, phase, output, hashes, provenance,
                 *, info=None, integrity=lambda: None):
    """Execute a single reviewed phase, using synthetic model injection in tests."""
    import torch

    config.validate()
    if phase not in {"development", "test"} or (phase == "test") != (info is not None):
        raise InputError("Choose development or a previously frozen baseline test.")
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Unreviewed metadata cannot enter the baseline.")
    if any(p.requires_grad for p in model.parameters()):
        raise InputError("The frozen control must have no trainable parameters.")
    if any(provenance.get(k) != v for k, v in MODEL.items()):
        raise InputError("Baseline provenance must identify the selected pinned encoder.")
    integrity()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise InputError("CUDA unavailable; automatic CPU fallback is prohibited.")
    output = Path(output).resolve()
    if output.exists():
        raise InputError("Baseline outputs require a new private attempt directory.")
    if phase == "test":
        # Reopen/replay authoritative artifacts, rather than trusting a caller's
        # in-memory threshold or selection. No development pixels are loaded.
        info = inspect_baseline(info["root"], images, pairs, config, hashes)
        if any(hashes.get(k) != value for k, value in info["hashes"].items()):
            raise InputError("Baseline artifacts differ from the reviewed frozen selection.")
        if any(info["record"]["provenance"].get(k) != provenance.get(k) for k in COMPATIBILITY):
            raise InputError("Baseline source/environment changed; review compatibility before final test.")
        if (output.is_relative_to(info["root"]) or info["root"].is_relative_to(output)
                or (info["root"] / "final_test_frozen_claim.json").exists()):
            raise InputError("Use a separate output; a claimed frozen final test cannot be repeated automatically.")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model.to(config.device).float()
    model.eval()
    integrity()
    output.mkdir(parents=True)
    if phase == "test":
        write_json(info["root"] / "final_test_frozen_claim.json", {
            "output": str(output), "input_sha256": hashes,
            "claimed_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        write_json(output / "run_record.json", {"schema_version": 1, "arm": "frozen_dinov2",
            "phase": phase, "config": asdict(config), "input_sha256": hashes, "provenance": provenance,
            "inference_precision": "fp32", "optimizer_created": False,
            "started_at_utc": datetime.now(timezone.utc).isoformat()})
        with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
            def event(row):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()

            selected, scores = score_partition(model, images, pairs, pixel_loader, config.device, event,
                                                partition=phase)
        integrity()
        write_json(output / (phase + "_scores.json"), [asdict(row) for row in scores])
        if phase == "development":
            # Every failure must be investigated; a failed selection leaves its
            # private raw outcomes/events but never installs a frozen selection.
            development = evaluate_development(images, pairs, scores, config.target_far)
            report = {"schema_version": 1, "scope": "Frozen encoder development only; no test scoring",
                      "development": development, "test_scores_accessed": False}
            selection = {"development_complete": True, "selection_frozen": True,
                         "test_scores_accessed": False, "threshold": development["threshold"],
                         "development": development, "record_sha256": digest(output / "run_record.json"),
                         "development_scores_sha256": digest(output / "development_scores.json")}
            integrity()
            write_json(output / "selection.json", selection)
            combined = scores
        else:
            cutoff = info["development"]["threshold"]["cutoff"]
            by_score = {s.pair_id: s for s in scores}
            by_id = {i.image_id: i for i in images}
            report = {"schema_version": 1, "scope": "Frozen encoder final test with unchanged development cutoff",
                "threshold": info["development"]["threshold"],
                "test": summarize([(p.label, by_score[p.pair_id]) for p in selected], cutoff),
                "by_hand_same_pooled_cutoff": {hand: summarize(
                    [(p.label, by_score[p.pair_id]) for p in selected if by_id[p.image_a].hand == hand], cutoff)
                    for hand in ("left", "right")},
                "development_selection_replayed": True, "test_used_for_selection": False,
                "limits": ["Private review flags require underlying acquisition/exposure/protocol evidence.",
                           "Failures remain rejects with coverage reported; failed impostors are not biometric rejection.",
                           "Test ROC/DET/EER are descriptive; no uncertainty interval or population low-FAR bound is claimed."]}
            combined = info["scores"] + scores
        write_scores(output / "scores.csv", combined)
        integrity()
        write_json(output / "report.json", report)
        return report
    except Exception:
        write_json(output / "failure.json", {"status": "failed", "phase": phase,
                   "claim_retained": phase == "test", "automatic_retry_allowed": False})
        raise
