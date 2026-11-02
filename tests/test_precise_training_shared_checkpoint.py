"""Synthetic checkpoint tensor verification; no real weights or images."""

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching.data import InputError  # noqa: E402
from verify_precise_shared_checkpoint import parameter_integrity  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("torch"), "Runs in the matcher CPU CI job")
class SharedCheckpointTests(unittest.TestCase):
    def fixture(self):
        import torch
        names = ("embeddings.weight", "encoder.layer.10.weight", "encoder.layer.11.weight", "layernorm.weight")
        return {name: torch.ones(2) for name in names}

    def test_selected_updates_and_unchanged_frozen_tensors_are_counted_exactly(self):
        base, adapted = self.fixture(), self.fixture()
        adapted["encoder.layer.10.weight"] += .1
        result = parameter_integrity(base, adapted)
        self.assertEqual(result["frozen_tensors"], 1)
        self.assertEqual(result["changed_frozen_tensors"], 0)
        self.assertEqual(result["trainable_tensors"], 3)
        self.assertEqual(result["changed_trainable_tensors"], 1)

    def test_changed_frozen_missing_extra_nonfinite_and_incompatible_tensors_fail(self):
        import torch
        for change in ("frozen", "missing", "extra", "nonfinite", "shape", "dtype"):
            with self.subTest(change=change):
                base, adapted = self.fixture(), self.fixture()
                if change == "frozen": adapted["embeddings.weight"] += 1
                elif change == "missing": adapted.pop("layernorm.weight")
                elif change == "extra": adapted["extra.weight"] = torch.ones(1)
                elif change == "nonfinite": adapted["layernorm.weight"][0] = float("nan")
                elif change == "shape": adapted["layernorm.weight"] = torch.ones(3)
                elif change == "dtype": adapted["layernorm.weight"] = torch.ones(2, dtype=torch.float64)
                with self.assertRaises(InputError):
                    parameter_integrity(base, adapted)

    def test_serialization_key_mismatch_is_not_mistaken_for_tensor_changes(self):
        base, adapted = self.fixture(), self.fixture()
        base["encoder.layer.10.attention.attention.query.weight"] = base.pop("encoder.layer.10.weight")
        with self.assertRaises(InputError):
            parameter_integrity(base, adapted)


if __name__ == "__main__":
    unittest.main()
