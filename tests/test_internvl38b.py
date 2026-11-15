"""Offline contracts: no model download, GPU, private dataset or real inference."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import internvl38b_runtime as iv  # noqa: E402


class InternVL38BTests(unittest.TestCase):
    def setUp(self):
        self.config = iv.load_config(iv.ROOT / "code/configs/internvl3_38b_offload.json")

    def test_strict_parser_never_accepts_errors_or_prose(self):
        for raw in ("A", " a ", "(A).", "[b]", "B."):
            self.assertIn(iv.parse_answer(raw), ("A", "B"))
        for raw in (None, "", "INVALID", "UNKNOWN", "__ERROR__ CUDA out of memory",
                    "I cannot determine whether these match.", "Both images look similar", "A or B", "AA", "(A]"):
            self.assertIsNone(iv.parse_answer(raw), raw)

    def test_config_rejects_implicit_fallback_or_unpinned_model(self):
        for changes in ({"revision": "0" * 40}, {"attention": "auto"}, {"quantization": "int8"},
                        {"cpu_dtype": "float32"}, {"max_tiles": 0}, {"max_tiles": 13},
                        {"gpu_reserve_gib": 0}, {"gpu_weight_budget_gib": True},
                        {"max_new_tokens": 999}, {"user_prompt": "<image> secret"}):
            with self.assertRaises(iv.PreflightError):
                replace(self.config, **changes).validate()

    def test_vision_chunking_preserves_values_and_order(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is optional for the chunking fixture")
        calls = []
        values = torch.arange(30).reshape(10, 3)
        def extractor(batch):
            calls.append(len(batch))
            return batch * 2 + 1
        result = iv.extract_in_chunks(extractor, values, 3)
        self.assertTrue(torch.equal(result, values * 2 + 1))
        self.assertEqual(calls, [3, 3, 3, 1])

    def test_snapshot_hashes_and_index_detect_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = json.dumps({"vision_model.weight": {
                "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
            shard = root / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"vision_model.weight": shard.name},
                                         "metadata": {"total_size": 8}}))
            receipt = root / "receipt.json"
            members = [{"file": p.name, "bytes": p.stat().st_size, "sha256": iv.digest(p)}
                       for p in (shard, index)]
            receipt.write_text(json.dumps({"repo_id": iv.REPO_ID, "revision": iv.REVISION,
                                          "files": members, "bytes": sum(r["bytes"] for r in members)}))
            self.assertEqual(iv.tensor_inventory(root)["vision_model.weight"]["elements"], 4)
            iv.verify_snapshot(root, receipt, full=True)
            data = shard.read_bytes(); shard.write_bytes(data[:-1] + b"1")
            with self.assertRaises(iv.PreflightError):
                iv.verify_snapshot(root, receipt, full=True)
            shard.write_bytes(data)
            index.write_text(index.read_text().replace('"total_size": 8', '"total_size": 9'))
            with self.assertRaises(iv.PreflightError):
                iv.tensor_inventory(root)

    def test_snapshot_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp); root = parent / "snapshot"; root.mkdir()
            outside = parent / "outside"; outside.write_bytes(b"data")
            receipt = parent / "receipt.json"
            receipt.write_text(json.dumps({"repo_id": iv.REPO_ID, "revision": iv.REVISION,
                "bytes": 4, "files": [{"file": "../outside", "bytes": 4, "sha256": iv.digest(outside)}]}))
            with self.assertRaises(iv.PreflightError):
                iv.verify_snapshot(root, receipt)

    def test_memory_plan_blocks_busy_host_but_does_not_change_device_map(self):
        specs = {f"language_model.model.layers.{i}.weight": {"elements": 100_000_000} for i in range(64)}
        specs.update({"vision_model.weight": {"elements": 1_000_000_000},
                      "mlp1.weight": {"elements": 10_000_000},
                      "language_model.model.embed_tokens.weight": {"elements": 100_000_000},
                      "language_model.lm_head.weight": {"elements": 100_000_000},
                      "language_model.model.norm.weight": {"elements": 1000}})
        config = replace(self.config, gpu_weight_budget_gib=3)
        enough = {"cuda_available": True, "gpu_free_bytes": 16 * iv.GIB,
                  "ram_available_bytes": 64 * iv.GIB}
        good = iv.memory_plan(specs, set(specs) - {"language_model.lm_head.weight"}, config, enough)
        self.assertTrue(good["ready_to_load"])
        busy = dict(enough, gpu_free_bytes=4 * iv.GIB, ram_available_bytes=iv.GIB)
        blocked = iv.memory_plan(specs, set(specs) - {"language_model.lm_head.weight"}, config, busy)
        self.assertFalse(blocked["ready_to_load"])
        self.assertEqual(blocked["device_map"], good["device_map"])
        self.assertEqual(set(blocked["blockers"]), {"insufficient_free_gpu_memory", "insufficient_available_system_ram"})
        self.assertEqual(good["device_map"]["language_model.model.layers.63"], 0)
        self.assertGreater(good["cpu_decoder_layers"], 0)

    def test_blocked_plan_never_loads_checkpoint(self):
        # Dependency imports are avoided by the synthetic dispatcher when capacity fails.
        import run_precise_internvl38b as cli
        with patch.object(cli, "load_weights", side_effect=AssertionError("must not load")):
            self.assertFalse(cli.synthetic(None, self.config, {"plan": {"ready_to_load": False}}, None))

    def test_tile_grid_retains_high_resolution_and_includes_thumbnail(self):
        self.assertEqual(iv.tile_grid(1600, 1500, 12, 448), (3, 3))
        self.assertEqual(iv.tile_grid(1600, 1500, 1, 448), (1, 1))

    def fixture_bundle(self, root):
        images, rows = {}, []
        for hand in ("left", "right", "thumbs"):
            for label in ("genuine", "impostor"):
                index = len(rows) + 1
                suffix = {"left": "14", "right": "13", "thumbs": "15"}[hand]
                a = f"{index}_a_{suffix}.jpeg"
                b = f"{index if label == 'genuine' else index + 10}_b_{suffix}.jpeg"
                for name in (a, b):
                    (root / name).write_bytes(b"synthetic")
                    images[name] = {"path": name, "sha256": hashlib.sha256(b"synthetic").hexdigest()}
                rows.append({"pair_id": str(index), "image_a": a, "image_b": b,
                             "hand": hand, "label": label, "numeric_identity_disagreement": False})
        bundle = {"scope": "numeric_group_disjoint_fixed_pairs", "labels": "as_supplied", "images": images,
                  "testing": rows, "training": [{"image_a": "99_a.jpeg", "image_b": "99_b.jpeg"}]}
        path = root / "bundle.json"; path.write_text(json.dumps(bundle))
        return path, bundle

    def test_cohort_enforces_membership_and_resolves_only_local_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path, bundle = self.fixture_bundle(root)
            rows, paths, report = iv.cohort(path, root, self.config, expected_per_cell=1)
            self.assertEqual(report["pairs"], 6); self.assertEqual(len(paths), 12)
            bundle["testing"][1] = dict(bundle["testing"][0])
            path.write_text(json.dumps(bundle))
            with self.assertRaises(iv.PreflightError):
                iv.cohort(path, root, self.config, expected_per_cell=1)
            self.assertEqual(rows[0]["label"], "genuine")

    def test_summary_does_not_credit_failed_impostor_as_correct(self):
        rows = [{"pair_id": "g", "hand": "left", "label": "genuine"},
                {"pair_id": "i", "hand": "left", "label": "impostor"}]
        outcomes = [{"pair_id": "g", "status": "ok", "answer": "A"},
                    {"pair_id": "i", "status": "execution_error", "answer": None}]
        report = iv.decision_summary(rows, outcomes)["pooled"]
        self.assertEqual(report["valid_confusion"], {"tp": 1, "tn": 0, "fp": 0, "fn": 0})
        self.assertEqual(report["accuracy_all_expected_failures_incorrect"], .5)
        self.assertEqual(report["valid_coverage"], .5)
        incomplete = iv.decision_summary(rows, outcomes[:1])["pooled"]
        self.assertIsNone(incomplete["accuracy_all_expected_failures_incorrect"])
        self.assertEqual(incomplete["unattempted"], 1)

    def test_review_needs_full_model_synthetic_pass_and_exact_human_binding(self):
        identity = "1" * 64; receipt = "2" * 64
        prior = {"mode": "synthetic", "status": "passed", "real_weights_loaded": True,
                 "fingerprint_inference": False, "identity_sha256": identity, "synthetic_passed": True,
                 "snapshot": {"verification": "full_sha256"}}
        binding = {"identity": identity, "bundle": iv.CURRENT_BUNDLE_SHA256, "synthetic_report": receipt}
        review = {"input_sha256": binding, "execution_authorized": True, "protocol_reviewed": True,
                  "human_response_reference": "synthetic test fixture only"}
        self.assertEqual(iv.evaluation_review(prior, review, identity, iv.CURRENT_BUNDLE_SHA256, receipt), binding)
        for changes in ({"mode": "probe"}, {"real_weights_loaded": False}, {"status": "blocked"},
                        {"identity_sha256": "3" * 64}, {"synthetic_passed": False}):
            with self.assertRaises(iv.PreflightError):
                iv.evaluation_review(dict(prior, **changes), review, identity, iv.CURRENT_BUNDLE_SHA256, receipt)
        with self.assertRaises(iv.PreflightError):
            iv.evaluation_review(prior, dict(review, human_response_reference=""), identity,
                                 iv.CURRENT_BUNDLE_SHA256, receipt)
        with self.assertRaises(iv.PreflightError):
            iv.evaluation_review(prior, review, identity, "wrong-cohort", receipt)

    def test_journal_preserves_failure_and_stops_without_retry(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is optional for this executor fixture")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); image = root / "synthetic.png"
            Image.new("RGB", (8, 8), "white").save(image)
            rows = [{"pair_id": str(i), "image_a": "a", "image_b": "b"} for i in range(3)]
            with patch("builtins.print"):
                def fail(a, b):
                    raise RuntimeError("synthetic CUDA failure")
                results = iv.execute_pairs(rows, {"a": image, "b": image}, self.config, root, fail)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "execution_error")
            self.assertIsNone(results[0]["answer"])
            events = [json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["started", "finished"])
            with self.assertRaises(FileExistsError):
                iv.execute_pairs(rows, {"a": image, "b": image}, self.config, root, fail)

    def test_outputs_refuse_public_tree_and_overwrite(self):
        with self.assertRaises(iv.PreflightError):
            iv.private_output(iv.ROOT / "reports/should-never-be-created")
        with tempfile.TemporaryDirectory() as temp, patch.object(iv, "ROOT", Path(temp)):
            existing = Path(temp) / "local/precise/existing"
            existing.mkdir(parents=True)
            with self.assertRaises(iv.PreflightError):
                iv.private_output(existing)
            created = iv.private_output(Path(temp) / "local/precise/new-attempt")
            self.assertTrue(created.is_dir())


if __name__ == "__main__":
    unittest.main()
