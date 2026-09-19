"""Recover the interrupted Precise Pixtral CSV, then resume all prompts."""
import csv
import io
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / 'results/precise/pixtral/latest/rb_pixtral_task_description_20260907_1924.csv'

def main():
    data = TARGET.read_bytes()
    lines = data.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(data.decode('utf-8'), newline=''), strict=True)
    header = next(reader)
    last_good = reader.line_num
    count = 0
    try:
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Incomplete CSV record')
            last_good = reader.line_num
            count += 1
    except (csv.Error, ValueError) as exc:
        backup = TARGET.with_suffix('.csv.interrupted-backup')
        if backup.exists():
            raise RuntimeError('Backup already exists; inspect before recovery') from exc
        shutil.copy2(TARGET, backup)
        TARGET.write_bytes(b''.join(lines[:last_good]))
        print(f'Recovered {count} complete rows; preserved original at {backup}: {exc}', flush=True)
    attempt = TARGET.with_suffix('.csv.attempt')
    if attempt.exists():
        backup = attempt.with_suffix('.attempt.interrupted-backup')
        if backup.exists():
            raise RuntimeError('Attempt backup already exists')
        attempt.rename(backup)
        print('Preserved interrupted attempt marker; retrying all unfinished pairs.', flush=True)
    subprocess.run([
        sys.executable, '-u', 'code/run_ridgebase_prompted.py',
        '--model', 'pixtral', '--pairs', 'results/precise/pairs_precise_eval.csv',
        '--results-dir', 'results/precise', '--resume',
    ], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
