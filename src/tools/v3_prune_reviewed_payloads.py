"""One-time, exact-three-file cleanup authorized 27 September 2026.

No recursive removal or dynamic target discovery. --apply requires a separately
reviewed manifest and repeats identity, recovery-source, and open-file checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess

ROOT=Path('.').resolve()
OUT=ROOT/'artifacts/improvement_v3/storage_payload_cleanup_20260927'
TARGETS=(
    (ROOT/'artifacts/improvement_v2/cloud/train_payload_v1.tar.gz','fleet_v1'),
    (ROOT/'artifacts/improvement_v2/cloud/submission_payload_v1.tar.gz','submission_fleet_v1'),
    (ROOT/'artifacts/improvement_v2/cpu/prepared_gate0_train_100k_v1/X.npy.gz',None),
)
PROTECTED=(
    Path('work/ce/sub_v12_c8_dense10_keych/matching_results.tsv'),
    Path('work/ce/sub_v12_c8_dense10_keych/candidate_pairs.tsv'),
    Path('work/ce/e5_cont1300k/model.pt'),
    ROOT/'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm',
    ROOT/'artifacts/improvement_v2/cpu/fit_gate0_200k_v1/enhanced.cbm',
    ROOT/'artifacts/improvement_v2/cpu/prepared_gate0_train_100k_v1/manifest.json',
)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def identity(p):
    s=p.lstat()
    if p.resolve()!=p or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
        raise ValueError('Target is linked, nonregular, or redirected: '+str(p))
    return dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,
                device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,sha256=sha(p))


def no_readers():
    cmd=['lsof','-Fpn','--',*[str(p) for p,_ in TARGETS]]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    if p.returncode!=1 or p.stdout.strip():
        raise RuntimeError('An open target or failed open-file check; no deletion allowed')


def verify_sources():
    reports=[]
    for path,fleet in TARGETS:
        item=identity(path)
        if fleet:
            sp=ROOT/'artifacts/improvement_v2/cloud'/fleet/'state.json'
            state=json.loads(sp.read_text())
            if not state.get('payload_uploaded') or state.get('bucket_deleted'):
                raise ValueError('No retained uploaded payload receipt')
            if item['bytes']!=state['payload_bytes'] or item['sha256']!=state['payload_sha256']:
                raise ValueError('Local payload differs from uploaded checksum')
            cmd=['aws','s3api','head-object','--profile','amazon-ml','--region',state['region'],
                 '--bucket',state['bucket'],'--key','payload-v1.tar.gz']
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
            if result.returncode:raise RuntimeError('S3 retained-object check failed; no deletion')
            head=json.loads(result.stdout)
            if head['ContentLength']!=item['bytes']:raise ValueError('Retained object size differs')
            item['recovery']=dict(type='existing_S3_payload',state_path=str(sp),object_key='payload-v1.tar.gz',
                content_length=head['ContentLength'],last_modified=head['LastModified'],etag=head.get('ETag'),
                caveat='HEAD verifies presence/size; checksum comes from prior verified upload/deploy receipt. Temporary cloud copy may be removed by later scheduled cleanup.')
        else:
            mp=path.parent/'manifest.json';m=json.loads(mp.read_text());source=Path(m['source'])
            if not m['complete'] or m['references']!=100000 or len(m['shards'])!=500:
                raise ValueError('Unexpected training matrix provenance')
            for shard in m['shards']:
                fp=source/'shards'/shard['shard']/'features.npz'
                if not fp.is_file() or sha(fp)!=shard['features_sha256']:
                    raise ValueError('Required reconstruction shard missing/changed: '+str(fp))
            item['recovery']=dict(type='regenerate_from_retained_feature_shards',source=str(source),
                shards_verified=500,manifest_sha256=sha(mp),
                caveat='No trash copy. Recompute using tools/v2_fit_cpu.py prepare into a new directory; old gzip restore command will no longer work.')
        reports.append(item)
    return reports


def run(apply=False):
    if Path(__file__).resolve().parents[1]!=ROOT:raise ValueError('Unexpected workspace')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():raise FileExistsError('Cleanup already completed')
    no_readers();reports=verify_sources();protected={str(p):sha(p) for p in PROTECTED}
    plan=dict(targets=reports,protected_sha256=protected,
              total_bytes=sum(v['bytes'] for v in reports),allocated_bytes=sum(v['allocated_bytes'] for v in reports))
    if not apply:
        write(OUT/'review.json',plan)
        print(json.dumps({k:v for k,v in plan.items() if k!='protected_sha256'},indent=2));return
    old=json.loads((OUT/'review.json').read_text())
    if plan!=old:raise ValueError('Reviewed cleanup inputs or recovery sources changed')
    no_readers();before=shutil.disk_usage(ROOT).free
    with (OUT/'deletion_journal.jsonl').open('x') as journal:
        for path,_ in TARGETS:
            want=next(x for x in reports if x['path']==str(path))
            current=identity(path)
            if any(current[k]!=want[k] for k in current):raise ValueError('Target changed immediately before unlink')
            path.unlink()
            journal.write(json.dumps(dict(path=str(path),bytes=want['bytes'],sha256=want['sha256']))+'\n');journal.flush()
    assert all(not p.exists() for p,_ in TARGETS)
    assert protected=={str(p):sha(p) for p in PROTECTED},'Protected artifact changed'
    done=dict(complete=True,utc=datetime.now(timezone.utc).isoformat(),deleted_files=len(TARGETS),
        deleted_bytes=plan['total_bytes'],allocated_bytes_released=plan['allocated_bytes'],free_before=before,
        free_after=shutil.disk_usage(ROOT).free,protected_sha256=protected,review_sha256=sha(OUT/'review.json'),
        no_recursive_removal=True,no_cloud_mutations=True,
        recovery='Two payloads currently retained in S3; compressed training matrix requires recomputation from 500 checksum-verified feature shards. None is in trash.')
    write(OUT/'completion.json',done);print(json.dumps(done,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');run(**vars(p.parse_args()))
