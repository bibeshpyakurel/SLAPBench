"""Reproduce newly supplied Qwen decisions on the sealed new-cohort test pairs.

This verifies saved outcomes and source consistency, not absent run provenance.
It does not relax the separate fully reviewed trained-Qwen comparison gate.
No inference, binary ROC/EER, test tuning or raw-input rewriting is performed.
"""

import argparse
import json
import hashlib
from pathlib import Path
import pickletools
import struct
import sys
import zipfile

from audit_precise_revised_cohort import TRAIN_JSON
from compare_precise_qwen_ab import operating_summary, read_ab
from precise_matching.data import InputError
from precise_matching.evaluation import Score
from precise_matching.fixed_pairs import GROUP_SCOPE, FixedConfig, FixedPair, validate_group_holdout
from precise_matching.training import digest, write_json
from report_precise_shared_pairs import ARMS, dependency_counts, read_dino

ROOT = Path(__file__).resolve().parents[1]


def summary_check(summary, pairs, decisions, prompt):
    if any(summary.get(k) != prompt[k] for k in ("system_prompt", "user_prompt")):
        raise InputError("Supplied summary prompt differs from the shared training source.")
    expected_hash = hashlib.sha256(json.dumps([prompt["system_prompt"], prompt["user_prompt"]], ensure_ascii=False).encode()).hexdigest()
    if summary.get("prompt_sha256") != expected_hash:
        raise InputError("Source prompt digest does not use its declared prompt bytes.")
    totals = operating_summary(pairs, decisions)
    counts = {}
    for label in ("genuine", "impostor"):
        rows = [decisions[p.pair_id] for p in pairs if p.label == label]
        counts[label] = {"attempted": len(rows), "predicted_a": sum(r.accept is True for r in rows),
            "predicted_b": sum(r.accept is False for r in rows),
            "invalid": sum(r.status in {"unknown_response", "invalid_response"} for r in rows),
            "errors": sum(r.status == "execution_error" for r in rows)}
        if summary.get(label + "_counts") != counts[label]:
            raise InputError("Supplied summary counters do not reproduce from preserved outcomes.")
    g, i = counts["genuine"], counts["impostor"]
    matrix = {"TP": g["predicted_a"], "FN": g["predicted_b"], "FP": i["predicted_a"], "TN": i["predicted_b"]}
    if summary.get("confusion_matrix") != matrix:
        raise InputError("Supplied confusion matrix does not reproduce.")
    attempted = len(pairs); correct = matrix["TP"] + matrix["TN"]
    accuracy = correct / attempted
    if summary["metrics"]["overall_accuracy_all_attempts_percent"] != round(100 * accuracy, 4):
        raise InputError("Supplied accuracy does not reproduce.")
    return {"confusion_matrix": matrix, "accuracy_all_attempts": accuracy, "operating_point": totals}


def paired_saved_decisions(pairs, dino_scores, qwen_decisions, cutoff):
    from precise_matching.comparison import paired_decisions
    dino = {s.pair_id: s for s in dino_scores}
    qwen = {p.pair_id: Score(p.pair_id, "ok" if qwen_decisions[p.pair_id].accept is not None else "error",
             float(qwen_decisions[p.pair_id].accept) if qwen_decisions[p.pair_id].accept is not None else None) for p in pairs}
    if len(dino) != len(dino_scores) or set(dino) != {p.pair_id for p in pairs}:
        raise InputError("Paired DINO outcomes must cover the exact complete testing membership.")
    return paired_decisions(pairs, dino, qwen, cutoff, .5)


def static_training_arguments(path):
    """Read primitive pickle opcodes only; never unpickle/execute supplied code."""
    wanted = {"num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
              "learning_rate", "seed", "max_steps", "load_best_model_at_end", "bf16", "fp16",
              "dataloader_drop_last", "warmup_steps"}
    with zipfile.ZipFile(path) as archive:
        data = archive.read(next(n for n in archive.namelist() if n.endswith("data.pkl")))
    ops = list(pickletools.genops(data)); found = {}
    for index, (_, key, _) in enumerate(ops):
        if not isinstance(key, str) or key not in wanted:
            continue
        for op, value, _ in ops[index + 1:]:
            if op.name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:continue
            if op.name in {"BININT", "BININT1", "BININT2", "BINFLOAT", "INT", "FLOAT"}:found[key] = value
            elif op.name in {"NEWTRUE", "NEWFALSE"}:found[key] = op.name == "NEWTRUE"
            elif op.name == "NONE":found[key] = None
            break
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "bundle", "config", "dino-run-root", "output", "private-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if (args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "reports")
                or args.private_receipt.exists() or not args.private_receipt.resolve().is_relative_to(ROOT / "local/precise")):
            raise InputError("Use new aggregate/private receipt paths.")
        sources = {"bundle": args.bundle, "config": args.config}
        sources.update({str(p.relative_to(args.intake)): p for p in args.intake.rglob("*") if p.is_file()})
        for short in ("frozen", "adapted"):
            for phase in ("calibration", "test"):
                directory = args.dino_run_root / (short + "-" + phase)
                sources.update({str(p.resolve()): p for p in directory.iterdir() if p.suffix in {".csv", ".json"}})
        hashes = {key: digest(path) for key, path in sources.items()}
        bundle = json.loads(args.bundle.read_text())
        if bundle.get("scope") != GROUP_SCOPE or bundle.get("labels") != "as_supplied":
            raise InputError("New Qwen audit requires the declared numeric-group cohort.")
        training, testing = ([FixedPair(**r) for r in bundle[k]] for k in ("training", "testing"))
        validate_group_holdout(training, testing)
        train_name = "training-source/" + TRAIN_JSON
        if hashes[train_name] != bundle["source_sha256"][TRAIN_JSON]:
            raise InputError("Newly supplied training JSON differs from the DINO cohort.")
        prompt = json.loads(sources[train_name].read_text()); qwen = {}; decisions = {}; source_claims = {}
        for kind in ("base", "finetuned"):
            summary = json.loads((args.intake / f"summary_{kind}_matched.json").read_text())
            if summary.get("mode") != kind:
                raise InputError("Unexpected source run mode.")
            paths = {label: args.intake / f"res_test_{label}_2100_nonzero_{kind}_matched.txt" for label in ("genuine", "impostor")}
            decisions[kind], tokens = read_ab(paths, testing)
            qwen[kind] = {**summary_check(summary, testing, decisions[kind], prompt), "source_token_counts": tokens,
                "by_slap_type": {hand: operating_summary([p for p in testing if p.hand == hand], decisions[kind])
                                 for hand in ("left", "right", "thumbs")}}
            source_claims[kind] = {key: summary[key] for key in ("base_model", "prompt_sha256", "inference")}
        config = FixedConfig(**json.loads(args.config.read_text())); dino = {}
        for arm in ARMS:
            short = arm.split("_")[0]
            data, scores = read_dino(args.dino_run_root / (short + "-test"), args.dino_run_root / (short + "-calibration"),
                                     training, testing, config, arm)
            record = json.loads((args.dino_run_root / (short + "-test/run_record.json")).read_text())
            if any(record["provenance"]["input_sha256"][k] != hashes[k] for k in ("bundle", "config")):
                raise InputError("DINO outcome belongs to a different cohort/configuration.")
            dino[arm] = {"threshold": data["threshold"], "code_commit": data["code_commit"],
                         "model_identity": data["model_identity"], "pooled": data["results"]["pooled"]["all_attempts_failure_as_reject"],
                         "by_slap_type": {hand: value["all_attempts_failure_as_reject"] for hand, value in data["results"]["by_slap_type"].items()}}
            # Supplemental same-attempt operating comparisons only; never treat A/B as score confidence.
            qwen["finetuned"].setdefault("paired_dino_to_qwen", {})[arm] = paired_saved_decisions(
                testing, scores, decisions["finetuned"], data["threshold"]["cutoff"])
        checkpoint = args.intake / "checkpoint-27897"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        with (checkpoint / "adapter_model.safetensors").open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_length < 10 * 1024**2:raise InputError("Unexpected adapter header length.")
            header = json.loads(stream.read(header_length))
        adapter_config = json.loads((checkpoint / "adapter_config.json").read_text())
        evidence = {"checkpoint_name": checkpoint.name, "adapter_sha256": hashes["checkpoint-27897/adapter_model.safetensors"],
            "adapter_config_sha256": hashes["checkpoint-27897/adapter_config.json"],
            "adapter_tensor_count": len([k for k in header if k != "__metadata__"]),
            "adapter_base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
            "adapter_base_revision": adapter_config.get("revision"),
            "trainer_state": {k: state[k] for k in ("epoch", "global_step", "max_steps", "num_train_epochs", "best_model_checkpoint")},
            "static_training_args": static_training_arguments(checkpoint / "training_args.bin"),
            "binary_training_config_supplied": (checkpoint / "binary_training_config.json").exists(),
            "checkpoint_digest_bound_at_evaluation": False}
        report = {"schema_version": 1, "scope": "source_verified_common_pair_decisions_run_provenance_incomplete",
            "benchmark_scope": GROUP_SCOPE, "outcome_membership_verified": True, "training_json_matches_dino_bundle": True,
            "training_pairs": len(training), "testing_pairs": len(testing), "dependence": dependency_counts(training, testing),
            "qwen": qwen, "dino": dino, "source_run_claims": source_claims, "checkpoint_evidence": evidence,
            "binary_qwen_roc_auc_eer": None, "source_sha256": {k: v for k, v in hashes.items() if k in {str(p.relative_to(args.intake)) for p in args.intake.rglob('*') if p.is_file()}},
            "provenance_limits": ["Exact pairs, supplied labels, parsed predictions, counts and summary accuracies are independently joined/reproduced.",
                "Training JSON equality and checkpoint-state consistency do not by themselves prove actual complete membership consumption.",
                "The summary references binary_training_config.json, but that actual file has not been supplied.",
                "Resolved base revisions and actual runtime/package versions are absent; the adapter's base identifier differs from the base summary identifier.",
                "The provided adapter is hashed now; no digest in the evaluation summaries proves those exact bytes produced the saved predictions.",
                "Raw generated replies/logs are absent; parsed A/B coverage does not independently establish original response-format compliance.",
                "The final checkpoint state is supplied, but a documented checkpoint-selection history and confirmation of no test-based selection remain absent.",
                "Qwen native binary decisions and DINO nominal-1%-training-FAR cutoffs are different operating policies, not matched test-FAR methods.",
                "No population confidence interval, significance, biological-identity/capture independence or causal module claim."]}
        if any(digest(path) != hashes[key] for key, path in sources.items()):raise InputError("A supplied input changed during audit.")
        args.private_receipt.parent.mkdir(parents=True, exist_ok=True); args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.private_receipt, {"input_sha256": hashes, "code_sha256": digest(Path(__file__)),
                   "no_model_inference": True, "no_pickle_execution": True, "no_threshold_selection": True})
        write_json(args.output, report)
    except (InputError, OSError, KeyError, ValueError, TypeError, struct.error, zipfile.BadZipFile):
        print("Qwen source audit did not complete; inspect private source evidence.", file=sys.stderr);return 1
    print("Qwen exact-pair outcomes and source summaries verified; run-provenance limits retained.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
