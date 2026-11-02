"""Reviewed V1-V6-only file retirement. Never removes directories or reports.

Run without --apply to freeze exact paths, checksums and protected dependencies.
--apply checks that plan again, verifies no open files, and unlinks only those
regular files. Hard-linked storage is counted once, only when its last link goes.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT=Path('.').resolve()
C=Path('work/ce')
OUT=ROOT/'artifacts/improvement_v3/v1_v6_blob_retirement_20260927'
A=ROOT/'artifacts'

OLD_OUTPUT_DIRS=[
    A/'duf-fel_full_output_v1', A/'duf-fel_bundle_v1/output',
    A/'v1_conflict_resolution_v1/submission/output',
    A/'duf-fel_full_output_v2_100k', A/'duf-fel_full_output_v2_200k',
    A/'duf-fel_bundle_v2_100k/output', A/'duf-fel_bundle_v2_200k/output',
    A/'duf-fel_full_output_v3_100k_owner_v1',A/'duf-fel_full_output_v3_200k_owner_v1',
    A/'duf-fel_full_output_v4_neural1300k_ops_v1',
    A/'duf-fel_full_output_v4_neural1300k_peerhalf_v1',
    A/'duf-fel_full_output_v4_neural1300k_cpu_upgrade_v1',
    A/'duf-fel_full_output_v4_neural400k_ops_v1',A/'duf-fel_full_output_v4_neural400k_ops_v2',
    A/'public_best_0976_20260926',A/'improvement_v3/v6_model_input_candidates_v1',
    C/'sub_v4_stack_all',C/'sub_v4_stack_usin_frold',C/'sub_v5_fr_nop',C/'sub_v6_cC_band',
    C/'sub_repro_p100k',C/'sub_smoke_partial',
]
OLD_NEURAL_EXPORTS=[A/('duf-fel_full_output_v4_'+s) for s in (
    'neural1300k_ops_v1','neural1300k_peerhalf_v1','neural1300k_cpu_upgrade_v1')]
FIXED={
    'improvement_v2/cascade/pretrained_minilm/pytorch_model.bin':'Downloadable unused MiniLM initialization',
    'improvement_v2/cascade/run_v1b/model/model.safetensors':'Retired MiniLM cascade; retraining required',
    'improvement_v3/transfer_pilot_v1/models/xlmr/model.safetensors':'Downloadable unused XLM-R initialization',
    'improvement_v3/transfer_pilot_v1/xlmr/training_checkpoint.pt':'Abandoned XLM-R optimizer/training state; no direct undo',
    'improvement_v3/transfer_pilot_v1/xlmr/trainable_weights.pt':'Abandoned partial XLM-R fit; retraining required',
    'improvement_v3/transfer_pilot_v1/e5_features/embeddings.npy':'Old frozen-E5 pilot embeddings; recompute',
    'improvement_v3/transfer_pilot_v1/contrastive_features/embeddings.npy':'Old contrastive pilot embeddings; recompute',
    'improvement_v3/transfer_pilot_v1/models/indic/transformer/indicxlit.pt':'Unused downloadable IndicXlit weights; current learned dictionary retained',
    'improvement_v3/transfer_pilot_v1/models/indic/official-indic-en-v1.0.zip':'Unused IndicXlit download archive',
    'improvement_v3/empty_name_specialist_v1/model/model.pt':'Rejected missing-address specialist; retraining required',
    'improvement_v3/empty_name_specialist_v1/model/resume.pt':'Rejected specialist optimizer state; no direct undo',
    'improvement_v3/structural_v1/test100k/claims.sqlite3':'Old 100k-only ownership export database; reconstruct from retained scores',
    'improvement_v3/owner200k_v1/test/claims.sqlite3':'Old 200k-only ownership export database; reconstruct from retained scores',
    'improvement_v3/v6_model_input_candidates_v1/assembly.sqlite3':'Old V6 candidate assembly database; reconstruct if needed',
    'duf-fel_submission_v2_200k.zip':'Obsolete CPU-only package; metadata/code retained, not V12 final release',
}
OLD_SCORE_SUFFIXES=(
    '.stack_cC_band.npy','.stack_cC_nocty_nf2.npy',
    '.stack_e5_cont1300k_nocty_nf2_nop.npy','.stack_e5_cont1300k_nocty.npy','.dtok.npy',
)
KEEP_CANDIDATE=A/'duf-fel_full_output_v3_100k_owner_v1/candidate_pairs.tsv'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def output(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')


def candidates():
    found={}
    def add(p,reason):
        if p==KEEP_CANDIDATE:return
        if p.exists():
            if p.resolve()!=p or not stat.S_ISREG(p.lstat().st_mode):
                raise ValueError('Unexpected linked/nonregular target: '+str(p))
            found[p]=reason
    for d in OLD_OUTPUT_DIRS:
        for name in ('matching_results.tsv','candidate_pairs.tsv'):
            add(d/name,'Superseded V1-V6/CPU-only output. Result reports retained; recompute to restore.')
    for rel,reason in FIXED.items():add(A/rel,reason)
    for d in OLD_NEURAL_EXPORTS:
        for p in (d/'cache').glob('*.sqlite3'):add(p,'Retired V4 export database; reconstruct from retained source scores')
        for p in (d/'scores').glob('*.npy'):add(p,'Retired V4 export score cache; actual V12 scratch scores retained')
    for p in (C/'test').glob('part_*.pkl.*'):
        if p.name.endswith(OLD_SCORE_SUFFIXES):add(p,'Obsolete V4/V5/V6-only stack score or token feature; recompute')
    return found


def protected():
    files={KEEP_CANDIDATE}
    for d in [C/'sub_v12_c8_dense10_keych',C/'sub_v11_c8_dense10',C/'sub_v10_c8_dense',
              C/'sub_v9_c8_fr090',C/'sub_v8_c4_fr090',C/'sub_v7_fr090',C/'sub_v7_fr060']:
        files.update(p for p in d.iterdir() if p.is_file())
    files.update(C.glob('train_*.pkl'))
    for directory in ('e5_cont1300k','e5_pilot400k','e5_round3'):
        files.add(C/directory/'model.pt')
    for p in (C/'test').glob('part_*.pkl*'):
        if p.name.endswith(('.pkl','.ops.npy','.nf.npy','.nf2.npy','.e5_cont1300k.npy',
                            '.c8.npy','.stack_c8_band.npy','.stack_c8_nocty_nf2.npy',
                            '.stack_e5_cont1300k_nocty_nf2.npy')):files.add(p)
    for rel in ('improvement_v3/transfer_pilot_v1/models/e5/model.safetensors',
                'improvement_v3/models/e5_base/model.safetensors',
                'duf-fel_full_output_v4_neural1300k_peerhalf_v1/reproduction/model.pt',
                'improvement_v3/neural_followup_round3_v1/model.pt',
                'improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm',
                'improvement_v2/cpu/fit_gate0_200k_v1/enhanced.cbm',
                'improvement_v3/structural_v1/text_scorer.json',
                'improvement_v3/structural_v1/owner_eval/policy.json'):
        files.add(A/rel)
    files.add(C/'translit_cnt.pkl')
    # Retain all descriptions, code, logs and reports in the reviewed old trees.
    roots=OLD_OUTPUT_DIRS+OLD_NEURAL_EXPORTS+[
        A/'improvement_v2/cascade',A/'improvement_v3/transfer_pilot_v1',
        A/'improvement_v3/empty_name_specialist_v1',A/'improvement_v3/v6_model_input_candidates_v1',
    ]
    for d in roots:
        for p in d.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.md','.txt','.log','.py','.toml','.yaml','.yml','.tsv'):
                if p.name not in ('matching_results.tsv','candidate_pairs.tsv'):files.add(p)
    files.update(ROOT.glob('*.md'))
    for p in files:
        if not p.is_file():raise FileNotFoundError('Required protected dependency: '+str(p))
    return files


def no_open_files(targets):
    wanted={(p.stat().st_dev,p.stat().st_ino) for p in targets}
    found=[]
    # Same-user /proc FD and mapping inventory; no target path data is modified.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            for fd in (proc/'fd').iterdir():
                try:
                    s=fd.stat()
                    if (s.st_dev,s.st_ino) in wanted:found.append((proc.name,'fd',str(fd)))
                except (FileNotFoundError,ProcessLookupError):pass
            for line in (proc/'maps').read_text().splitlines():
                parts=line.split(None,5)
                if len(parts)<5:continue
                major,minor=(int(v,16) for v in parts[3].split(':'))
                if (os.makedev(major,minor),int(parts[4])) in wanted:found.append((proc.name,'mapping'))
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            # Restricted unrelated programs are also checked by lsof below.
            continue
    import subprocess
    paths=list(targets)
    for start in range(0,len(paths),150):
        r=subprocess.run(['lsof','-Fpn','--',*map(str,paths[start:start+150])],capture_output=True,text=True,timeout=45)
        if r.returncode not in (0,1):raise RuntimeError('Open-file check failed')
        if r.stdout.strip():found.append(('lsof','open_target'))
    if found:raise RuntimeError('A target inode is open; no deletion: '+str(found[:8]))


def reviewed():
    target=candidates();keep=protected()
    if set(target)&keep:raise ValueError('Protected-file overlap')
    no_open_files(target)
    rows=[];hashcache={}
    def checksum(p):
        s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if key not in hashcache:hashcache[key]=sha(p)
        return hashcache[key]
    for p,reason in sorted(target.items()):
        s=p.lstat()
        rows.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,
            device=s.st_dev,inode=s.st_ino,links=s.st_nlink,mtime_ns=s.st_mtime_ns,sha256=checksum(p),reason=reason))
    groups=defaultdict(list)
    for row in rows:groups[row['device'],row['inode']].append(row)
    released=sum(rs[0]['allocated_bytes'] for rs in groups.values() if len(rs)==rs[0]['links'])
    return dict(targets=rows,protected_sha256={str(p):checksum(p) for p in sorted(keep)},
                file_paths=len(rows),apparent_bytes=sum(r['bytes'] for r in rows),
                estimated_allocated_bytes_released=released,
                retained_shared_inodes=sum(len(rs)<rs[0]['links'] for rs in groups.values()))


def run(apply):
    if Path(__file__).resolve().parents[1]!=ROOT:raise ValueError('Unexpected workspace')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').exists():raise FileExistsError('Already completed')
    plan=reviewed()
    if not apply:
        output(OUT/'review.json',plan)
        print(json.dumps({k:v for k,v in plan.items() if k not in ('targets','protected_sha256')},indent=2))
        print('Review: '+str(OUT/'review.json'));return
    if plan!=json.loads((OUT/'review.json').read_text()):raise ValueError('Reviewed inputs changed; nothing removed')
    no_open_files([Path(r['path']) for r in plan['targets']]);free_before=shutil.disk_usage(ROOT).free
    with (OUT/'deletion_journal.jsonl').open('x') as journal:
        for row in plan['targets']:
            p=Path(row['path']);s=p.lstat()
            if p.resolve()!=p or not stat.S_ISREG(s.st_mode) or (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=(row['device'],row['inode'],row['bytes'],row['mtime_ns']):
                raise ValueError('Target changed before deletion: '+str(p))
            p.unlink();journal.write(json.dumps(row)+'\n');journal.flush()
    for p,h in plan['protected_sha256'].items():
        if sha(Path(p))!=h:raise ValueError('Protected artifact changed: '+p)
    done=dict(complete=True,utc=datetime.now(timezone.utc).isoformat(),deleted_paths=len(plan['targets']),
        apparent_bytes_removed=plan['apparent_bytes'],estimated_allocated_bytes_released=plan['estimated_allocated_bytes_released'],
        free_before=free_before,free_after=shutil.disk_usage(ROOT).free,
        protected_files_verified=len(plan['protected_sha256']),review_sha256=sha(OUT/'review.json'),
        no_directories_removed=True,no_cloud_changes=True,
        recovery='Not in trash. Downloadable initial weights can be fetched again; learned weights need retraining; outputs/caches need recomputation. Retained hard links remain readable.')
    output(OUT/'completion.json',done);print(json.dumps(done,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');run(**vars(p.parse_args()))
