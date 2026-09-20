"""Check the distinction between publishable findings and local-only assets."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("publication_scope", ROOT / "scripts/check_publication_scope.py")
scope = module_from_spec(spec)
spec.loader.exec_module(scope)


class PublicationScopeTests(unittest.TestCase):
    def test_local_assets_cannot_be_committed(self):
        for path in [
            "datasets/sd302b/image.png", "dataset/participants.csv",
            "models/qwen3vl-8b/config.json", "venv/bin/python",
            ".claude/settings.json", ".env", ".env.local",
            "related_papers/full_text.txt", "weights.safetensors",
            "archive.zip", "model.tar.gz", "id_rsa.key",
        ]:
            with self.subTest(path=path):
                self.assertIsNotNone(scope.exclusion_reason(path, 10))

    def test_research_outputs_remain_publishable(self):
        for path in [
            "code/run_verification.py", "results/precise/scores.csv",
            "results/precise/run.csv.interrupted-backup", "logs/run.log",
            "reports/response-audit-20260919/SUMMARY.md",
            "paper/manuscript/figures/fig_roc_curves.png", ".env.example",
        ]:
            with self.subTest(path=path):
                self.assertIsNone(scope.exclusion_reason(path, 10))

    def test_large_files_are_rejected(self):
        self.assertIsNone(scope.exclusion_reason("results/pairs.csv", scope.MAX_BYTES))
        self.assertIsNotNone(scope.exclusion_reason("results/pairs.csv", scope.MAX_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
