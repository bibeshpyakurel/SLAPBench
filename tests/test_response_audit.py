"""Regression cases for offline parsing; no inference or network access."""
import importlib.util
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit', ROOT / 'code/audit_saved_responses.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ParsingTests(unittest.TestCase):
    def test_clear_answers(self):
        for raw, expected in [('A', 'A'), ('(B)', 'B'), ('No, different individuals', 'B'),
                              ('Yes, same person', 'A'), ('No', 'B')]:
            with self.subTest(raw=raw):
                self.assertEqual(audit.parse_binary(raw)[0], expected)

    def test_no_word_letter_or_negation_guessing(self):
        for raw in ['Both images have ridges', 'Yes, different persons', 'No, same person',
                    'No, not sure', 'I cannot determine whether these match',
                    '__ERROR__ bad request', 'same?', 'not the same person', 'A or B']:
            with self.subTest(raw=raw):
                self.assertIsNone(audit.parse_binary(raw)[0])

    def test_numeric_scores(self):
        for raw, expected in [('95', 95), ('Score: 0', 0), ('100/100', 100), ('62.5', 62.5)]:
            self.assertEqual(audit.parse_score(raw)[0], expected)
        for raw in ['101', '-1', 'There are 4 fingers', '50 or 60', '__ERROR__ HTTP 429',
                    'cannot determine: 50', '', '1. Explain the images', '0 to 100', 'NaN']:
            with self.subTest(raw=raw):
                self.assertIsNone(audit.parse_score(raw)[0])

    def test_nul_repair_requires_exact_id_and_metadata(self):
        expected = dict(pair_id='G_test', label='mated', category='primary_genuine',
                        subject1='x', subject2='x', hand1='14', hand2='14')
        row = dict(expected, pair_id='\0\0G_test')
        self.assertEqual(audit.normalize_id(row, {'G_test': expected}),
                         ('G_test', 'nul_prefix_removed_exact_metadata'))
        row['hand1'] = '13'
        self.assertEqual(audit.normalize_id(row, {'G_test': expected})[1], 'unmatched')
        row['pair_id'] = '\0G_unknown'
        self.assertEqual(audit.normalize_id(row, {'G_test': expected})[1], 'unmatched')

    def test_existing_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / 'keep.txt'
            sentinel.write_text('existing analysis')
            result = subprocess.run(
                [sys.executable, '-B', str(ROOT / 'code/audit_saved_responses.py'),
                 '--output-dir', tmp], capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Output directory exists', result.stderr)
            self.assertEqual(sentinel.read_text(), 'existing analysis')
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])

    def test_real_exports_and_detect_tampering(self):
        fingerprints = {}
        result = audit.validate_exports(fingerprints)
        self.assertEqual(sum(v['rows'] for v in result['exports'].values()), 1352)
        self.assertEqual(result['id_repairs_in_memory_only'][0]['prefix_nul_bytes'], 612)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in fingerprints:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            target = root / 'results/precise/qwen3vl/res_precise_676_genuine_qwen3_vl-8b.txt'
            lines = target.read_text().splitlines()
            fields = lines[0].split()
            fields[2] = str(float(fields[2]) + 1)
            lines[0] = ' '.join(fields)
            target.write_text('\n'.join(lines)+'\n')
            with patch.object(audit, 'ROOT', root), self.assertRaises(ValueError):
                audit.validate_exports({})


if __name__ == '__main__':
    unittest.main()
