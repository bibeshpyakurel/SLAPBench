"""Offline, append-only audit of Qwen3 exports and saved prompted responses.

No model loads or API calls. Parsers below implement a conservative analysis
policy, not a replacement for historical inference parsers. Unrecognized prose
is withheld for review; labels are never inputs to parsing.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'conservative-v1'
REFUSAL = re.compile(r"\b(?:cannot|can't|unable to|don't know|not sure|cannot determine|can't determine)\b", re.I)
NUMBER = r'(?:\d+(?:\.\d+)?)'


def precheck(raw):
    text = raw.strip()
    if not text:
        return text, 'empty'
    if text.startswith('__ERROR__'):
        return text, 'error'
    if REFUSAL.search(text):
        return text, 'uncertainty_or_refusal_review'
    return text, None


def parse_binary(raw):
    text, reason = precheck(raw)
    if reason:
        return None, reason
    # Only full-string forms; A in "same" or B in "both" is never an answer.
    option = re.fullmatch(r'(?:answer\s*[:=]\s*)?\(?([AB])\)?[.!]?', text, re.I)
    if option:
        return option[1].upper(), 'explicit_option'
    text = text.lower().strip(' .!')
    yes = r'(?:yes|same|same person|same people|same individual|genuine|yes[,\s]+(?:same(?: person| people| individual)?|they do))'
    no = r'(?:no|different|different (?:person|persons|people|individuals)|impostor|no[,\s]+different(?: (?:person|persons|people|individuals))?)'
    if re.fullmatch(yes, text):
        return 'A', 'explicit_text'
    if re.fullmatch(no, text):
        return 'B', 'explicit_text'
    if (re.search(r'\byes\b', text) and re.search(r'\b(?:no|different)\b', text)) or (
        re.search(r'\bno\b', text) and re.search(r'\bsame\b', text)
    ):
        return None, 'contradictory_review'
    return None, 'unrecognized_review'


def parse_score(raw):
    text, reason = precheck(raw)
    if reason:
        return None, reason
    match = re.fullmatch(
        rf'(?:(?:similarity(?: score)?|matching score|score)\s*[:=]\s*)?({NUMBER})(?:\s*/\s*100)?[.]?',
        text, re.I,
    )
    if match:
        value = float(match[1])
        if 0 <= value <= 100:
            return value, 'explicit_score'
        return None, 'out_of_range'
    if not re.search(r'\d', text):
        return None, 'no_numeric_score'
    return None, 'numeric_prose_review'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path, fingerprints):
    fingerprints[str(path.relative_to(ROOT))] = sha(path.read_bytes())
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'Malformed CSV record in {path}')
        return rows


def normalize_id(row, manifest):
    original = row['pair_id']
    if original in manifest:
        return original, 'exact'
    candidate = original.lstrip('\x00')
    if candidate != original and candidate in manifest:
        keys = ('label', 'category', 'subject1', 'subject2', 'hand1', 'hand2')
        if all(row.get(k) == manifest[candidate].get(k) for k in keys):
            return candidate, 'nul_prefix_removed_exact_metadata'
    return original.replace('\x00', '<NUL>'), 'unmatched'


def validate_exports(fingerprints):
    base = ROOT / 'results/precise'
    pairs = read_csv(base / 'pairs_precise_eval.csv', fingerprints)
    manifest = {r['pair_id']: r for r in pairs}
    if len(manifest) != len(pairs):
        raise ValueError('Duplicate manifest IDs')
    path = base / 'qwen3vl/latest/rb_qwen3vl_similarity_score_20260906_1552.csv'
    scores = read_csv(path, fingerprints)
    mapped = {}
    repairs = []
    for index, row in enumerate(scores, 1):
        pid, status = normalize_id(row, manifest)
        if status == 'unmatched' or pid in mapped:
            raise ValueError('Unmatched or duplicate score ID')
        if status != 'exact':
            repairs.append(dict(source_record=index, corrected_pair_id=pid,
                                prefix_nul_bytes=len(row['pair_id']) - len(pid), method=status))
        for key in ('label', 'category', 'subject1', 'subject2', 'hand1', 'hand2'):
            if row[key] != manifest[pid][key]:
                raise ValueError(f'Score metadata mismatch: {key}')
        mapped[pid] = row
    results = {}
    for category, filename in [('primary_genuine', 'res_precise_676_genuine_qwen3_vl-8b.txt'),
                               ('primary_impostor', 'res_precise_676_impostore_qwen3_vl-8b.txt')]:
        expected = []
        for row in pairs:
            if row['category'] != category:
                continue
            score = mapped[row['pair_id']]
            value = float(score['similarity_score'])
            raw_value, _ = parse_score(score['raw_response'])
            if raw_value != value:
                raise ValueError('Raw response and stored score differ')
            expected.append((Path(row['img1_path']).name, Path(row['img2_path']).name, value))
        export = base / 'qwen3vl' / filename
        data = export.read_bytes()
        fingerprints[str(export.relative_to(ROOT))] = sha(data)
        actual = []
        for line in data.decode().splitlines():
            fields = line.split()
            if len(fields) != 3:
                raise ValueError('Export format mismatch')
            actual.append((fields[0], fields[1], float(fields[2])))
        if len(expected) != 676 or actual != expected or len(set(actual)) != 676:
            raise ValueError('Export rows/order/values/uniqueness mismatch')
        results[filename] = dict(status='PASS', rows=676, filename_pairs_scores_order_match=True,
                                scores_match_raw_responses=True)
    missing = [r for r in pairs if r['pair_id'] not in mapped]
    return dict(exports=results, id_repairs_in_memory_only=repairs,
                missing_categories=dict(Counter(r['category'] for r in missing)),
                limitation='Validates saved-file consistency, not image identity or acquisition provenance.')


def auc(rows):
    from sklearn.metrics import roc_auc_score
    genuine = [score for category, score in rows if category == 'primary_genuine']
    impostor = [score for category, score in rows if category == 'primary_impostor']
    return dict(genuine=len(genuine), impostor=len(impostor),
                auc=float(roc_auc_score([1]*len(genuine)+[0]*len(impostor), genuine+impostor))
                if genuine and impostor else None)


def audit_file(path, out, fingerprints, manifest):
    rows = read_csv(path, fingerprints)
    score_mode = 'similarity_score' in rows[0]
    field = 'similarity_score' if score_mode else 'llm_answer'
    reasons = Counter()
    stored_missing = Counter()
    changes = Counter()
    groups = defaultdict(Counter)
    stored_scores, derived_scores = [], []
    examples = defaultdict(list)
    records = []
    for index, row in enumerate(rows, 1):
        raw = row.get('raw_response', '')
        value, reason = parse_score(raw) if score_mode else parse_binary(raw)
        original = row.get(field, '')
        old = float(original) if score_mode and original.strip() else original or None
        if score_mode and isinstance(old, float) and not math.isfinite(old):
            old = None
        pid, id_status = normalize_id(row, manifest) if manifest else (row['pair_id'], 'not_audited')
        reasons[reason] += 1
        if score_mode and old is None:
            stored_missing[reason] += 1
        elif not score_mode and old == 'INVALID':
            stored_missing[reason] += 1
        if value != old:
            changes['total'] += 1
            changes['definitive_reclassification' if value is not None and old not in (None, 'INVALID') else
                    'recovered' if value is not None else 'withheld_for_review'] += 1
            if len(examples[reason]) < 3:
                examples[reason].append(dict(record=index, stored=old, derived=value, raw=raw[:240]))
        category = row.get('category') or ('primary_genuine' if row['label'] == 'genuine' else 'primary_impostor')
        if score_mode:
            if old is not None:
                stored_scores.append((category, old))
            if value is not None:
                derived_scores.append((category, value))
        else:
            counts = groups[category]
            counts['rows'] += 1
            counts['valid'] += value is not None
            counts['accepts'] += value == 'A'
            counts['rejects'] += value == 'B'
            counts['abstains'] += value is None
            counts['stored_accepts'] += old == 'A'
        records.append(dict(source_record=index, pair_id=pid, id_status=id_status,
                            stored=original, derived='' if value is None else value,
                            parse_status=reason, raw_sha256=sha(raw.encode())))
    rel = path.relative_to(ROOT / 'results')
    dest = out / 'parsed' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    metrics = {}
    for category, counts in groups.items():
        metrics[category] = dict(counts)
        metrics[category]['acceptance_given_valid'] = counts['accepts']/counts['valid'] if counts['valid'] else None
        metrics[category]['acceptance_over_all_rows'] = counts['accepts']/counts['rows']
        metrics[category]['coverage'] = counts['valid']/counts['rows']
    result = dict(source=str(path.relative_to(ROOT)), rows=len(rows), mode='score' if score_mode else 'binary',
                  parse_status=dict(reasons), previously_missing_by_reason=dict(stored_missing),
                  changes=dict(changes), examples=dict(examples), binary_by_category=metrics)
    if score_mode:
        result['stored_score_subset'] = auc(stored_scores)
        result['conservative_score_subset'] = auc(derived_scores)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        ap.error('Output directory exists; choose a new path to preserve prior reports')
    fingerprints = {}
    exports = validate_exports(fingerprints)
    out.mkdir(parents=True)
    manifests = {}
    for dataset in ('precise', 'ridgebase'):
        path = ROOT / f'results/{dataset}/pairs_{dataset}_eval.csv'
        manifests[dataset] = {r['pair_id']: r for r in read_csv(path, fingerprints)}
    paths = sorted(p for dataset in ('precise', 'ridgebase', 'sd302b')
                   for p in (ROOT / 'results' / dataset).rglob('*.csv')
                   if ('latest' in p.parts or 'previous' in p.parts) and
                   any(strategy in p.name for strategy in ('similarity_score', 'zero_shot', 'task_description')))
    runs = [audit_file(p, out, fingerprints, manifests.get(p.relative_to(ROOT / 'results').parts[0])) for p in paths]
    for name, digest in fingerprints.items():
        if sha((ROOT / name).read_bytes()) != digest:
            raise RuntimeError('An input changed during analysis; report is not final')
    report = dict(parser_policy=VERSION, export_validation=exports, runs=runs,
                  source_sha256=fingerprints, audit_script_sha256=sha(Path(__file__).read_bytes()),
                  limitations=['Conservative full-string grammar; withheld prose needs human review.',
                               'Subset AUC is descriptive and may be selection-biased; no imputed scores.',
                               'Binary diagnostic impostors are reported separately.',
                               'Source record is 1-based CSV data record, not physical line number.'])
    (out / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Validated 1,352 exports; audited {len(runs)} prompted CSVs; sources unchanged. Report: {out}')


if __name__ == '__main__':
    main()
