"""Check the distinction between publishable findings and local-only assets."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
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
            "local/precise/metadata/source.csv", "local/precise/crops/crop.png",
            "local/precise/checkpoints/config.json", "checkpoints/model/config.json",
            "runs/precise/raw_scores.csv", "wandb/run/config.yaml",
            "mlruns/meta.yaml", "lightning_logs/version_0/hparams.yaml",
            "tensorboard/events.json",
        ]:
            with self.subTest(path=path):
                self.assertIsNotNone(scope.exclusion_reason(path, 10))

    def test_workspace_and_training_outputs_are_ignored_by_git(self):
        paths = [
            "local/precise/metadata/source.csv", "local/precise/splits/subjects.json",
            "local/precise/manifests/pairs.csv", "local/precise/minutiae/template.xyt",
            "local/precise/crops/crop.png", "local/precise/runs/raw_scores.csv",
            "local/precise/checkpoints/config.json", "checkpoints/config.json",
            "runs/raw_scores.csv", "wandb/config.yaml", "mlruns/meta.yaml",
            "lightning_logs/hparams.yaml", "tensorboard/events.json",
        ]
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT,
            input="\n".join(paths) + "\n", capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), set(paths))

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
