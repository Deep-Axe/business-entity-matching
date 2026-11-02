"""Explicitly retire generated 50k baseline caches authorized by the user.

Preserve original datasets, reference-selection ledgers, fitted models/reports,
final historical submissions, every newer cache, and shared hard-linked files.
Review and apply are separate; apply deletes only verified manifest entries.
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess

from v3_transfer_data import ROOT, digest, write
from v3_neural_test_pipeline import read

OUT=ROOT/'artifacts/improvement_v3/retired_50k_caches_v1'
PREPARED=('artifacts/improvement_v2/cpu/prepared_train_50k',
          'artifacts/improvement_v2/cpu/prepared_gate0_train_50k_v1')
FEATURE_ROOTS=('artifacts/deadline_features_train_50k_v1',
    'artifacts/improvement_v2/training/gate0_train_50k_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_0_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_1_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_2_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_3_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_4_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_5_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_6_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_features_7_v1')
PREDICTION_ROOTS=('artifacts/deadline_test_predictions_local_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_0_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_1_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_2_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_3_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_4_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_5_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_6_v1',
    'artifacts/aws_baseline_20260925_v1/collected/test_predictions_7_v1')
PREPARED_NAMES={'X.npy','X.npy.gz','y.npy','query_ids.npy','reference_ids.npy','true_counts.npy','countries.npy'}
PREDICTION_NAMES={'candidate_pairs.tsv','matching_results.tsv','probabilities.npy','pair_ids.npz'}


def allowed(relative):
    p=Path(relative)
    return ((str(p.parent) in PREPARED and p.name in PREPARED_NAMES)
        or (str(p.parent.parent.parent) in FEATURE_ROOTS and p.parent.parent.name=='shards'
            and p.parent.name.isdigit() and p.name=='features.npz')
        or (str(p.parent.parent.parent) in PREDICTION_ROOTS and p.parent.parent.name=='shards'
            and p.parent.name.isdigit() and p.name in PREDICTION_NAMES))


def no_open_files(paths):
    for start in range(0,len(paths),250):
        p=subprocess.run(['fuser',*map(str,paths[start:start+250])],capture_output=True,text=True,timeout=30)
        if p.returncode!=1 or p.stdout.strip() or p.stderr.strip():
            raise ValueError('A reviewed cache is open or the open-file check failed: '+p.stdout[-500:]+p.stderr[-500:])


def preservation():
    old=ROOT/'artifacts/duf-fel_full_output_v1';assembly=read(old/'assembly.json')
    if not assembly['complete_submission'] or assembly['references']!=1732544:raise ValueError('Old final submission incomplete')
    bound={str(old/name):h for name,h in assembly['output_sha256'].items()}
    bound[str(ROOT/'artifacts/public_best_0976_20260926/matching_results.tsv')]='5a0be554e60f88e763de95a5b4cc8f30f7cee352067ba24c16947e6319752118'
    for p,h in bound.items():
        if digest(p)!=h:raise ValueError('Preserved submission changed')
    for folder in FEATURE_ROOTS+PREDICTION_ROOTS:
        config=read(ROOT/folder/'config.json')
        if folder in PREDICTION_ROOTS and config['model_sha256']!=assembly['model_sha256']:
            raise ValueError('Prediction cache is not the retired 50k model')
        if folder.startswith('artifacts/aws_baseline_') and config['gate_terms']!=1:
            raise ValueError('Old feature-cache retrieval recipe differs')
    return bound


def review():
    OUT.mkdir(parents=True,exist_ok=False);bound=preservation();paths=[];shared=[]
    for folder in PREPARED:
        paths.extend(p for p in (ROOT/folder).iterdir() if p.name in PREPARED_NAMES)
    for folder in FEATURE_ROOTS:
        paths.extend(p for p in (ROOT/folder/'shards').rglob('features.npz'))
    for folder in PREDICTION_ROOTS:
        paths.extend(p for p in (ROOT/folder/'shards').rglob('*') if p.name in PREDICTION_NAMES)
    entries=[];totals=Counter()
    for path in sorted(set(paths)):
        relative=str(path.relative_to(ROOT))
        if not allowed(relative) or not path.is_file() or path.is_symlink() or path.resolve()!=path:
            raise ValueError('Unexpected retirement target: '+relative)
        s=path.stat()
        if s.st_nlink!=1:shared.append(relative);continue
        entries.append(dict(path=relative,size=s.st_size,inode=s.st_ino,mtime_ns=s.st_mtime_ns,sha256=digest(path)))
        totals[str(path.relative_to(ROOT).parent.parent)]+=s.st_size
    no_open_files([ROOT/e['path'] for e in entries])
    write(OUT/'review.json',dict(code_sha256=digest(__file__),files=entries,expected_bytes=sum(e['size'] for e in entries),
        preserved_shared_links=shared,preserved_submission_hashes=bound,original_dataset_removed=False,
        note='Exact allowlist of derived 50k training/old inference caches. Keep small models, metadata and exclusion reference ledgers.'))
    print('Reviewed',len(entries),'files;',sum(e['size'] for e in entries)/1024**3,'GiB; shared links preserved',len(shared),flush=True)


def apply():
    review=read(OUT/'review.json')
    if review['code_sha256']!=digest(__file__):raise ValueError('Reviewed removal code changed')
    preservation();files=review['files'];before=shutil.disk_usage(ROOT).free
    if any(not allowed(e['path']) for e in files):raise ValueError('Manifest escaped explicit allowlist')
    for e in files:
        path=ROOT/e['path'];s=path.stat()
        if (path.is_symlink() or path.resolve()!=path or s.st_nlink!=1 or s.st_ino!=e['inode']
                or s.st_mtime_ns!=e['mtime_ns'] or s.st_size!=e['size'] or digest(path)!=e['sha256']):
            raise ValueError('Cache changed after review: '+e['path'])
    no_open_files([ROOT/e['path'] for e in files])
    for i,e in enumerate(files):
        path=ROOT/e['path'];s=path.stat()
        if s.st_ino!=e['inode'] or s.st_mtime_ns!=e['mtime_ns'] or s.st_nlink!=1:raise ValueError('Cache changed before unlink')
        path.unlink()
        if (i+1)%500==0:write(OUT/'progress.json',dict(removed_files=i+1,expected_files=len(files)))
    for p,h in review['preserved_submission_hashes'].items():
        if digest(p)!=h:raise ValueError('Preserved output changed during retirement')
    write(OUT/'completion.json',dict(complete=True,removed_files=len(files),removed_bytes=sum(e['size'] for e in files),
        free_bytes_before=before,free_bytes_after=shutil.disk_usage(ROOT).free,review_sha256=digest(OUT/'review.json'),
        original_dataset_removed=False,models_removed=False,newer_caches_removed=False,submissions_removed=False,
        recovery='Removed derived files are not in trash; recomputation from preserved original data is required. Final old predictions remain available.'))
    print('Retired',len(files),'generated cache files; originals and final submissions preserved.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['review','apply']);a=p.parse_args()
    review() if a.action=='review' else apply()
