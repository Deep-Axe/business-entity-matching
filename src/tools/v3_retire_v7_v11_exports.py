"""Retire reviewed historical exports; preserve V12 and shared model inputs.

Without --apply: record exact files, hashes, identities and protected assets.
With --apply: require an identical review and no open target files before unlink.
No recursive deletion, remote changes, model deletion, or source edits.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat

from v3_retire_v1_v6_blobs import ROOT, A, C, sha, output, no_open_files

OUT = A / 'improvement_v3/v7_v11_export_retirement_20260927'
OLD = (
    'sub_v7_fr060', 'sub_v7_fr090', 'sub_v8_c4_fr090',
    'sub_v9_c8_fr090', 'sub_v10_c8_dense', 'sub_v11_c8_dense10',
)
SUFFIXES = ('.stack_c4_band.npy', '.stack_c4_nocty_nf2.npy')
V12 = C / 'sub_v12_c8_dense10_keych'
V12_HASHES = {
    'matching_results.tsv': 'a146a2bd177865817c50e9a339300bd96d132b09660a25cccb897922461f5946',
    'candidate_pairs.tsv': '1b432d65d68516ad5cda5adc751811e24443255789803dc1e7c7a29ffd388da7',
}


def inspect():
    targets = {}
    reports = {}
    for name in OLD:
        d = C / name
        reports[name] = json.loads((d / 'report.json').read_text())
        for filename in ('matching_results.tsv', 'candidate_pairs.tsv'):
            targets[d / filename] = 'Superseded export; report retained, rebuild needed to restore.'
    for suffix in SUFFIXES:
        found = sorted((C / 'test').glob('part_*.pkl' + suffix))
        if len(found) != 128:
            raise ValueError('Unexpected C4 score-shard count: ' + suffix)
        for p in found:
            targets[p] = 'V8-only final stack score; raw C4 ensemble scores and recipes retained.'
    for p in targets:
        if p.resolve() != p or not stat.S_ISREG(p.lstat().st_mode):
            raise ValueError('Nonregular/linked target: ' + str(p))
    keep = set(V12.iterdir())
    keep.update(C / name / 'report.json' for name in OLD)
    # All other saved inference inputs are protected, including V10 sources
    # reused by dense_test_rows2.py to construct the deployed K10 candidate set.
    for directory in ('test', 'test_dense', 'test_dense_k10', 'test_keych'):
        keep.update(p for p in (C / directory).rglob('*') if p.is_file() and p not in targets)
    for directory in ('e5_cont1300k', 'e5_pilot400k', 'e5_round3'):
        keep.add(C / directory / 'model.pt')
    keep.add(C / 'translit_cnt.pkl')
    for rel in (
        'improvement_v3/structural_v1/text_scorer.json',
        'improvement_v3/structural_v1/owner_eval/policy.json',
        'improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm',
        'improvement_v2/cpu/fit_gate0_200k_v1/enhanced.cbm',
    ):
        keep.add(A / rel)
    if keep & targets.keys():
        raise ValueError('Protected target overlap')
    no_open_files(targets)
    rows = []
    for p, reason in sorted(targets.items()):
        s = p.lstat()
        rows.append(dict(path=str(p), bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                         device=s.st_dev, inode=s.st_ino, links=s.st_nlink,
                         mtime_ns=s.st_mtime_ns, sha256=sha(p), reason=reason))
    hashes = {str(p): sha(p) for p in sorted(keep)}
    for name, expected in V12_HASHES.items():
        if hashes[str(V12 / name)] != expected:
            raise ValueError('V12 differs from verified submitted artifact: ' + name)
    for name, report in reports.items():
        row = next(r for r in rows if r['path'] == str(C / name / 'matching_results.tsv'))
        if row['sha256'] != report['matching_sha256']:
            raise ValueError('Historical matching/report mismatch: ' + name)
    groups = defaultdict(list)
    for row in rows:
        groups[row['device'], row['inode']].append(row)
    released = sum(rs[0]['allocated_bytes'] for rs in groups.values() if len(rs) == rs[0]['links'])
    return dict(targets=rows, protected_sha256=hashes, reports=reports,
                allocated_bytes_to_release=released,
                apparent_bytes=sum(r['bytes'] for r in rows))


def run(apply=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'completion.json').exists():
        raise FileExistsError('Already completed')
    plan = inspect()
    if not apply:
        output(OUT / 'review.json', plan)
        print(json.dumps(dict(files=len(plan['targets']),
                              protected_files=len(plan['protected_sha256']),
                              allocated_bytes_to_release=plan['allocated_bytes_to_release'],
                              review=str(OUT / 'review.json')), indent=2))
        return
    if plan != json.loads((OUT / 'review.json').read_text()):
        raise ValueError('Reviewed inputs changed; no deletion')
    no_open_files([Path(r['path']) for r in plan['targets']])
    before = shutil.disk_usage(ROOT).free
    with (OUT / 'deletion_journal.jsonl').open('x') as journal:
        for row in plan['targets']:
            p = Path(row['path'])
            s = p.lstat()
            identity = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            expected = (row['device'], row['inode'], row['bytes'], row['mtime_ns'])
            if p.resolve() != p or not stat.S_ISREG(s.st_mode) or identity != expected:
                raise ValueError('Target changed before deletion: ' + str(p))
            p.unlink()
            journal.write(json.dumps(row) + '\n')
            journal.flush()
    for path, expected in plan['protected_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Protected asset changed: ' + path)
    done = dict(complete=True, utc=datetime.now(timezone.utc).isoformat(),
                deleted_paths=len(plan['targets']),
                allocated_bytes_released=plan['allocated_bytes_to_release'],
                free_before=before, free_after=shutil.disk_usage(ROOT).free,
                protected_files_verified=len(plan['protected_sha256']),
                v12_sha256=V12_HASHES, no_directories_removed=True,
                no_models_removed=True, no_cloud_changes=True,
                recovery='Not in trash. Retired TSVs and C4 stack scores need recomputation; reports and hashes remain.')
    output(OUT / 'completion.json', done)
    print(json.dumps(done, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    run(**vars(parser.parse_args()))
