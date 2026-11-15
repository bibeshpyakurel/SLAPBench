"""Prepare InternVL3-38B offloading; real scoring requires a bound synthetic pass and review."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time

from internvl38b_runtime import (ROOT, PreflightError, append_event, canonical_digest, cohort,
                                decision_summary, digest, evaluation_review, execute_pairs, load_config, load_weights,
                                memory_plan, model_skeleton, now, predict, private_output, resources,
                                runtime_identity, tensor_inventory, tiny_offload_probe, tile_grid,
                                validate_context, verify_snapshot, write_json)


def prepare(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_MODULES_CACHE"] = str(ROOT / "local/precise/cache/internvl38b-modules")
    config = load_config(args.config)
    import torch
    torch.set_num_threads(config.cpu_threads)
    snapshot = verify_snapshot(args.model_dir, args.snapshot_receipt)
    if snapshot["files"] != 36:
        raise PreflightError("The full 36-file official snapshot is required.")
    runtime = runtime_identity()
    specs = tensor_inventory(args.model_dir)
    model, tokenizer, linear = model_skeleton(str(args.model_dir), config)
    if {name: list(p.shape) for name, p in model.named_parameters()} != {
            name: row["shape"] for name, row in specs.items()}:
        raise PreflightError("Meta model parameters differ from the verified weight index.")
    plan = memory_plan(specs, linear, config, resources())
    grid = tile_grid(1600, 1500, config.max_tiles, config.image_size)
    patch_count = grid[0] * grid[1] + int(config.thumbnail and grid != (1, 1))
    _, input_tokens = validate_context(tokenizer, model, config, [patch_count, patch_count])
    preprocessing = {"synthetic_size": [1600, 1500], "grid": list(grid),
                     "patch_counts": [patch_count, patch_count], "input_tokens": input_tokens,
                     "output_token_budget": config.max_new_tokens,
                     "tokenizer_limit": tokenizer.model_max_length}
    del model, tokenizer
    identity = {"config": asdict(config), "snapshot_receipt_sha256": digest(args.snapshot_receipt),
                "runtime": runtime, "device_map": plan["device_map"]}
    return config, {"schema_version": 1, "utc": now(), "identity": identity,
                    "identity_sha256": canonical_digest(identity), "snapshot": snapshot, "plan": plan,
                    "synthetic_context_check": preprocessing,
                    "real_weights_loaded": False, "fingerprint_inference": False}


def synthetic(args, config, report, output):
    if not report["plan"]["ready_to_load"]:
        return False
    import torch
    from PIL import Image, ImageDraw
    # Full checksum verification is repeated before any checkpoint tensor load.
    report["snapshot"] = verify_snapshot(args.model_dir, args.snapshot_receipt, full=True)
    if runtime_identity() != report["identity"]["runtime"]:
        raise PreflightError("Runtime/source changed after planning.")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats()
    write_json(output / "load-intent.json", {"utc": now(), "identity": report["identity_sha256"]})
    model, tokenizer = load_weights(str(args.model_dir), config, report["plan"])
    report["real_weights_loaded"] = True
    report["placement"] = model._slapbench_placement
    write_json(output / "loaded.json", {"utc": now(), "identity": report["identity_sha256"],
                                        "placement": report["placement"],
                                        "torch_gpu_allocated_bytes": torch.cuda.memory_allocated()})
    images = []
    for shift in (0, 37):
        image = Image.new("RGB", (1600, 1500), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, 1500, 100):
            draw.rectangle((x + shift, 200, x + shift + 20, 1200), fill="black")
        images.append(image)
    responses = []
    with (output / "synthetic-events.jsonl").open("x") as stream:
        for index in range(2):
            append_event(stream, {"event": "started", "call": index, "utc": now()})
            response = predict(model, tokenizer, config, *images)
            responses.append(response)
            append_event(stream, {"event": "finished", "call": index, **response})
    passed = all(r["answer"] in {"A", "B"} for r in responses)
    passed = passed and responses[0]["raw_response"] == responses[1]["raw_response"]
    report.update(synthetic_passed=passed, responses=responses,
                  peak_torch_gpu_bytes=torch.cuda.max_memory_allocated(),
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if runtime_identity() != report["identity"]["runtime"]:
        raise PreflightError("Runtime/source changed during synthetic inference.")
    return passed


def evaluate(args, config, report, output):
    if any(value is None for value in (args.bundle, args.image_root, args.synthetic_report, args.review)):
        raise PreflightError("Evaluation requires bundle, image root, successful synthetic report and review.")
    prior = json.loads(args.synthetic_report.read_text())
    review = json.loads(args.review.read_text())
    binding = evaluation_review(prior, review, report["identity_sha256"],
                                digest(args.bundle), digest(args.synthetic_report))
    if not report["plan"]["ready_to_load"]:
        raise PreflightError("Resources do not satisfy the fixed synthetic-tested configuration.")
    rows, paths, audit = cohort(args.bundle, args.image_root, config, verify_images=True)
    # Persistent single-use claim survives process failure; never re-score a completed attempt automatically.
    claim_key = {key: binding[key] for key in ("identity", "bundle")}
    claim = ROOT / "local/precise/metadata/internvl38b-claims" / (canonical_digest(claim_key) + ".json")
    claim.parent.mkdir(parents=True, exist_ok=True)
    write_json(claim, {"utc": now(), "binding": binding, "output": str(output)})
    report["cohort"] = audit
    report["snapshot"] = verify_snapshot(args.model_dir, args.snapshot_receipt, full=True)
    if runtime_identity() != report["identity"]["runtime"]:
        raise PreflightError("Runtime/source changed after review.")
    import torch
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    write_json(output / "load-intent.json", {"utc": now(), "identity": report["identity_sha256"]})
    model, tokenizer = load_weights(str(args.model_dir), config, report["plan"])
    report["real_weights_loaded"] = True
    report["placement"] = model._slapbench_placement
    write_json(output / "inputs.json", {"identity": report["identity"], "binding": binding,
                                        "review_sha256": digest(args.review), "cohort": audit})
    outcomes = execute_pairs(rows, paths, config, output,
                             lambda a, b: predict(model, tokenizer, config, a, b))
    report.update(fingerprint_inference=True, operating_points=decision_summary(rows, outcomes),
                  completed=len(outcomes) == len(rows) and
                  not any(r["status"] == "execution_error" for r in outcomes))
    if digest(args.bundle) != binding["bundle"] or runtime_identity() != report["identity"]["runtime"]:
        raise PreflightError("Cohort/runtime/source changed during execution; results are not sealed.")
    cohort(args.bundle, args.image_root, config, verify_images=True)
    return report["completed"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "plan", "synthetic", "evaluate"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-receipt", type=Path,
                        default=ROOT / "reports/internvl3-38b-download-20261115/snapshot.json")
    parser.add_argument("--config", type=Path, default=ROOT / "code/configs/internvl3_38b_offload.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--synthetic-report", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    output = private_output(args.output)
    start = time.monotonic()
    report = {"schema_version": 1, "mode": args.mode, "started_utc": now(),
              "real_weights_loaded": False, "fingerprint_inference": False}
    try:
        if args.mode == "probe":
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            config = load_config(args.config)
            import torch
            torch.set_num_threads(config.cpu_threads)
            report["runtime"] = runtime_identity()
            report["probe"] = tiny_offload_probe(output, config)
            passed = report["probe"]["passed"]
        else:
            config, prepared = prepare(args)
            report.update(prepared)
            write_json(output / "preflight.json", report)
            if args.mode == "plan":
                passed = report["plan"]["ready_to_load"]
            elif args.mode == "synthetic":
                passed = synthetic(args, config, report, output)
            else:
                passed = evaluate(args, config, report, output)
        report["status"] = "passed" if passed else "blocked"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    report.update(finished_utc=now(), elapsed_seconds=time.monotonic() - start)
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "blockers": report.get("plan", {}).get("blockers", []),
                      "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
