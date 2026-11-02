"""Retire two verified, regenerable leftovers from the old CPU experiments."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile

from v3_retire_v1_v6_blobs import ROOT, A, C, sha, output, no_open_files

OUT=A/'improvement_v3/v1_v6_tail_cache_retirement_20260927'
MATRIX=A/'improvement_v2/cpu/prepared_gate0_train_20k_v1/X.npy'
TAR=A/'improvement_v2/cloud/train200k_tail_v1.tar'


def inspect():
    targets=(MATRIX,TAR);no_open_files(targets)
    mpath=MATRIX.parent/'manifest.json';m=json.loads(mpath.read_text())
    assert m['complete'] and m['references']==20000 and len(m['shards'])==100
    assert sha(ROOT/'tools/v2_fit_cpu.py')==m['runner_sha256']
    assert sha(ROOT/'tools/v2_cpu_features.py')==m['feature_code_sha256']
    for shard in m['shards']:
        f=Path(m['source'])/'shards'/shard['shard']/'features.npz'
        assert sha(f)==shard['features_sha256'],str(f)
    receipt=json.loads((TAR.parent/'train200k_tail_v1.verification.json').read_text())
    assert receipt['complete'] and sha(TAR)==receipt['sha256']
    retained=A/'improvement_v2/training/gate0_train_200k_v1';count=0
    with tarfile.open(TAR) as archive:
        for member in archive:
            if not member.isfile():raise ValueError('Unexpected nonregular member')
            p=retained/member.name
            if not p.resolve().is_relative_to(retained.resolve()):raise ValueError('Unsafe member path')
            assert p.is_file() and p.stat().st_size==member.size
            with p.open('rb') as f,archive.extractfile(member) as g:
                while True:
                    a=f.read(1048576);b=g.read(1048576)
                    assert a==b,'Retained tail source mismatch: '+str(p)
                    if not a:break
            count+=1
    assert count==receipt['members']
    rows=[]
    for p in targets:
        s=p.lstat()
        assert p.resolve()==p and p.is_file() and s.st_nlink==1
        rows.append(dict(path=str(p),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,
                         mtime_ns=s.st_mtime_ns,sha256=sha(p),allocated_bytes=s.st_blocks*512))
    protected=[mpath,MATRIX.parent/'reference_ids.npy',
        A/'improvement_v2/cpu/fit_gate0_20k_v1/enhanced.cbm',
        C/'sub_v12_c8_dense10_keych/matching_results.tsv',
        C/'sub_v12_c8_dense10_keych/candidate_pairs.tsv']
    return dict(targets=rows,matrix_source_shards_verified=100,archive_source_members_byte_compared=count,
                protected_sha256={str(p):sha(p) for p in protected})


def run(apply):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():raise FileExistsError('Already retired')
    plan=inspect()
    if not apply:output(OUT/'review.json',plan);print(json.dumps(plan,indent=2));return
    assert plan==json.loads((OUT/'review.json').read_text()),'Reviewed sources changed'
    no_open_files((MATRIX,TAR));before=shutil.disk_usage(ROOT).free
    with (OUT/'deletion_journal.jsonl').open('x') as f:
        for row in plan['targets']:
            p=Path(row['path']);s=p.stat()
            assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])
            p.unlink();f.write(json.dumps(row)+'\n');f.flush()
    assert all(sha(Path(p))==h for p,h in plan['protected_sha256'].items())
    done=dict(complete=True,utc=datetime.now(timezone.utc).isoformat(),deleted_paths=2,
        deleted_bytes=sum(r['bytes'] for r in plan['targets']),free_before=before,free_after=shutil.disk_usage(ROOT).free,
        retained_sources_verified=True,protected_files_verified=True,
        recovery='Not in trash. Rebuild the 20k matrix from retained 100 feature shards; rearchive retained tail sources.')
    output(OUT/'completion.json',done);print(json.dumps(done,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');run(**vars(p.parse_args()))
