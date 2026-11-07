"""Training-only calibration and sealed held-out-pair evaluation."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .data import InputError
from .encoders import cosine_scores, encode
from .evaluation import Score, load_scores, select_cutoff, summarize, write_scores
from .fixed_pairs import GROUP_SCOPE, SHARED_SCOPE, SCOPES, canonical_digest, validate_group_holdout, validate_pairs
from .training import digest, write_json


def score_fixed(model, pairs, pixel_loader, device, batch_images=8, *, event=lambda row: None):
    """One outcome per pair, preserving decode/encoder failures explicitly."""
    import torch

    validate_pairs(pairs)
    if type(batch_images) is not int or batch_images < 1:
        raise InputError("A positive inference batch size is required.")
    needed = sorted({key for p in pairs for key in (p.image_a, p.image_b)})
    vectors, failures = {}, {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(needed), batch_images):
            keys, pixels = [], []
            for key in needed[start:start + batch_images]:
                try:
                    pixels.append(pixel_loader(key))
                    keys.append(key)
                except (InputError, OSError, ValueError):
                    failures[key] = "decode_error"
            if keys:
                try:
                    encoded = encode(model, torch.stack(pixels).to(device)).cpu()
                    vectors.update(zip(keys, encoded, strict=True))
                except (InputError, RuntimeError):
                    # No smaller-batch retry/fallback changes this declared attempt.
                    failures.update((key, "encoder_error") for key in keys)
            event({"images_attempted": min(start + batch_images, len(needed)),
                   "images_total": len(needed), "image_failures": len(failures)})
        scores = []
        for p in pairs:
            status = failures.get(p.image_a) or failures.get(p.image_b)
            if status:
                scores.append(Score(p.pair_id, status, None))
            else:
                value = float(cosine_scores(vectors[p.image_a][None], vectors[p.image_b][None])[0])
                scores.append(Score(p.pair_id, "ok", value))
    return scores


def matched_scores(pairs, scores):
    validate_pairs(pairs)
    by_score = {s.pair_id: s for s in scores}
    if len(by_score) != len(scores) or set(by_score) != {p.pair_id for p in pairs}:
        raise InputError("Outcomes must cover every supplied pair exactly once.")
    for s in scores:
        if ((s.status == "ok") != (s.score is not None)
                or s.status not in {"ok", "decode_error", "encoder_error", "missing_input", "error"}
                or s.score is not None and not math.isfinite(s.score)):
            raise InputError("Invalid outcome; failed comparisons require an empty score.")
    return by_score


def fixed_summary(pairs, scores, cutoff):
    if not math.isfinite(cutoff):
        raise InputError("A finite frozen cutoff is required.")
    by_score = matched_scores(pairs, scores)

    def subset(selected):
        report = summarize([(p.label, by_score[p.pair_id]) for p in selected], cutoff)
        for key in ("valid_scores_only", "all_attempts_failure_as_reject"):
            rates = report[key]
            rates["balanced_accuracy"] = ((rates["tar"] + 1 - rates["far"]) / 2
                                           if rates["tar"] is not None and rates["far"] is not None else None)
            rates["genuine_rejects"] = rates["genuine_comparisons"] - rates["genuine_accepts"]
            rates["impostor_rejects"] = rates["impostor_comparisons"] - rates["false_accepts"]
        return report

    return {"pooled": subset(pairs),
            "by_slap_type": {hand: subset([p for p in pairs if p.hand == hand])
                             for hand in ("right", "left", "thumbs")},
            "label_ambiguity_sensitivity": {
                "excluded_pairs": sum(p.numeric_identity_disagreement for p in pairs),
                "policy": "descriptive_exclusion_only_no_relabeling_or_retuning",
                "remaining": subset([p for p in pairs if not p.numeric_identity_disagreement])},
            "uncertainty": {"interpretation": "finite_cohort_descriptive_dependent_pairs",
                "independent_pair_intervals": False, "population_low_far_claim": False,
                "impostor_far_resolution": (1 / sum(p.label == "impostor" for p in pairs)),
                "reason": "Images and identities recur; no reviewed independent sampling model."}}


def calibrate_fixed(model, pairs, pixel_loader, config, output, provenance, *,
                    integrity=lambda: None, event=lambda row: None):
    """This API accepts only training pairs. Failures cannot relax the FAR budget."""
    output = Path(output).resolve()
    if output.exists():
        raise InputError("Calibration requires a new private directory.")
    output.mkdir(parents=True)
    write_json(output / "run_record.json", {"phase": "training_calibration", "config": asdict(config),
        "pairs_sha256": canonical_digest([asdict(p) for p in pairs]), "provenance": provenance})
    try:
        integrity()
        scores = score_fixed(model, pairs, pixel_loader, config.device, config.batch_pairs, event=event)
        write_scores(output / "scores.csv", scores)
        if any(s.status != "ok" for s in scores):
            raise InputError("Training calibration failures require investigation, not exclusions.")
        cutoff = select_cutoff([s.score for p, s in zip(pairs, scores, strict=True) if p.label == "impostor"],
                               config.target_far)
        cutoff["selected_on"] = "training_impostors_resubstitution"
        integrity()
        selection = {"schema_version": 1, "selection_frozen": True,
            "checkpoint_rule": config.checkpoint_rule, "threshold_rule": config.threshold_rule,
            "threshold": cutoff, "test_scores_accessed": False,
            "scores_sha256": digest(output / "scores.csv"),
            "run_record_sha256": digest(output / "run_record.json"),
            "model_identity": provenance["model_identity"],
            "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "selection.json", selection)
        return selection
    except BaseException as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__,
            "selection_frozen": False, "test_scores_accessed": False})
        raise


def inspect_calibration(calibration, training, config, provenance):
    """Replay the selection from immutable training outcomes before test access."""
    calibration = Path(calibration).resolve()
    selection = json.loads((calibration / "selection.json").read_text())
    record = json.loads((calibration / "run_record.json").read_text())
    if (selection.get("selection_frozen") is not True or selection.get("test_scores_accessed") is not False
            or selection.get("model_identity") != provenance["model_identity"]
            or record.get("config") != asdict(config)
            or record.get("pairs_sha256") != canonical_digest([asdict(p) for p in training])
            or record.get("provenance") != provenance
            or selection.get("scores_sha256") != digest(calibration / "scores.csv")
            or selection.get("run_record_sha256") != digest(calibration / "run_record.json")
            or selection.get("checkpoint_rule") != config.checkpoint_rule
            or selection.get("threshold_rule") != config.threshold_rule):
        raise InputError("Calibration provenance/selection changed.")
    scores = load_scores(calibration / "scores.csv")
    by_score = matched_scores(training, scores)
    if any(s.status != "ok" for s in scores):
        raise InputError("Calibration cannot silently omit failed training comparisons.")
    cutoff = select_cutoff([by_score[p.pair_id].score for p in training if p.label == "impostor"], config.target_far)
    cutoff["selected_on"] = "training_impostors_resubstitution"
    if selection["threshold"] != cutoff:
        raise InputError("Frozen threshold does not replay from training scores.")
    return selection


def test_fixed(model, training, testing, pixel_loader, config, calibration, output, provenance, *,
               integrity=lambda: None, event=lambda row: None, scope=SHARED_SCOPE):
    """Claim the sealed selection once before loading any test pixels."""
    output, calibration = Path(output).resolve(), Path(calibration).resolve()
    if scope not in SCOPES or provenance.get("benchmark_scope", SHARED_SCOPE) != scope:
        raise InputError("Test scope must match the reviewed benchmark provenance.")
    if scope == GROUP_SCOPE:
        validate_group_holdout(training, testing)
    integrity()
    selection = inspect_calibration(calibration, training, config, provenance)
    if output.exists():
        raise InputError("Evaluation requires a new private output directory.")
    write_json(calibration / "test_claim.json", {"output": str(output),
        "selection_sha256": digest(calibration / "selection.json"),
        "testing_pairs_sha256": canonical_digest([asdict(p) for p in testing]),
        "claimed_at_utc": datetime.now(timezone.utc).isoformat()})
    output.mkdir(parents=True)
    write_json(output / "run_record.json", {"phase": "held_out_pair_test", "provenance": provenance,
        "selection_sha256": digest(calibration / "selection.json"), "config": asdict(config),
        "testing_pairs_sha256": canonical_digest([asdict(p) for p in testing])})
    try:
        scores = score_fixed(model, testing, pixel_loader, config.device, config.batch_pairs, event=event)
        write_scores(output / "scores.csv", scores)
        integrity()
        # Guard against any mutation of the selection while inference ran.
        if inspect_calibration(calibration, training, config, provenance) != selection:
            raise InputError("Frozen calibration changed during testing.")
        group_holdout = scope == GROUP_SCOPE
        report = {"schema_version": 1, "scope": ("held_out_numeric_prefix_groups_identity_semantics_unconfirmed"
                  if group_holdout else "held_out_pairs_seen_images_and_identities"),
            "threshold": selection["threshold"], "model_identity": provenance["model_identity"],
            "results": fixed_summary(testing, scores, selection["threshold"]["cutoff"]),
            "limits": ["Agreement with supplied labels; numeric identity ambiguities are not corrected.",
                ("Training/testing numeric-prefix groups and images are disjoint; biological identity semantics remain unconfirmed."
                 if group_holdout else "All testing images and identity groups also appear in training."),
                "Threshold calibrated on training impostors, not independent development data.",
                "Descriptive test curves/EER must not select checkpoints, cutoffs or hyperparameters.",
                "No population confidence, low-FAR guarantee or acquisition independence is established."]}
        write_json(output / "aggregate.json", report)
        write_json(output / "completion.json", {"complete": True,
            "scores_sha256": digest(output / "scores.csv"), "aggregate_sha256": digest(output / "aggregate.json"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        return report
    except BaseException as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__,
            "test_claim_retained": True, "automatic_retry_authorized": False})
        raise
