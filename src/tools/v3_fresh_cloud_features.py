"""Prepare fresh-owner features after cloud extraction, without fitting a model.

Preserves frozen 25k fit / 5k calibration roles. Labels are transported, not
used to choose features, rows, weights, thresholds or a model. Memory-bounded
CPU work and cooperatively serialized GPU inference only; no test changes.
"""
from collections import Counter
import fcntl
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from ber.data import read_tsv
from ber.store import connect
from v3_neural_name_counts import Counts, matrix as structural_matrix
from v3_neural_test_pipeline import SCRATCH, STRUCT, read, resources, gpu_processes, feature_functions, edit_matrix
from v3_transfer_data import ROOT, digest, write

WORK=ROOT/'artifacts/improvement_v3/fresh_scale_features_v1'
CLOUD=ROOT/'artifacts/improvement_v3/fresh_cloud25k_v1'
FRESH=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'
ROLES=ROOT/'artifacts/improvement_v3/fresh_scale_roles_v1'
COUNTS=ROOT/'artifacts/improvement_v3/train_s1_structural_counts_v1'


def role_masks(owner_ids,roles):
    names=np.asarray([roles[str(o)] for o in owner_ids])
    fit=np.isin(names,['fit_existing','fit_new']);cal=names=='cal_new'
    if not np.all(fit^cal):raise ValueError('Unknown or overlapping owner roles')
    return fit,cal


def run():
    WORK.mkdir(parents=True,exist_ok=True);lock=(WORK/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+6*3600
    def progress(stage,**kw):write(WORK/'progress.json',dict(stage=stage,model_fitted=False,ready_for_submission=False,**kw))
    role_manifest=read(ROLES/'manifest.json')
    if not role_manifest['complete'] or digest(ROLES/'owner_roles.tsv')!=role_manifest['owner_roles_sha256']:
        raise ValueError('Frozen role manifest changed')
    checkpoints=dict(parent=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1/reproduction/model.pt',
                     round3=ROOT/'artifacts/improvement_v3/neural_followup_round3_v1/model.pt')
    core_path=Path('src/neural/name_unique_lib.py');functions=feature_functions()
    identity=dict(code_sha256=digest(__file__),role_manifest_sha256=digest(ROLES/'manifest.json'),
        plan_sha256=digest(CLOUD/'plan.json'),checkpoint_sha256={k:digest(p) for k,p in checkpoints.items()},
        core_sha256=digest(core_path),edit_source_sha256=functions[2],
        structural_source_sha256=digest(ROOT/'tools/v3_structural_features.py'),
        scorer_sha256=digest(ROOT/'tools/v3_ce_half_score.py'),
        fitting_performed=False,calibration_labels_used_for_selection=False)
    if (WORK/'identity.json').exists():
        if read(WORK/'identity.json')!=identity:raise ValueError('Frozen fresh-feature identity changed')
    else:write(WORK/'identity.json',identity)
    if (WORK/'completion.json').exists():return
    while not (CLOUD/'controller/completion.json').exists():
        if time.monotonic()>=end:raise TimeoutError('Finite cloud feature waiting window expired')
        progress('waiting_for_complete_cloud_extraction');time.sleep(30)
    resources();cloud=read(CLOUD/'controller/completion.json');fresh=read(FRESH/'completion.json')
    if (not cloud['complete'] or cloud['owners']!=25000 or cloud['plan_sha256']!=identity['plan_sha256']
            or cloud['role_manifest_sha256']!=identity['role_manifest_sha256'] or not fresh['complete']):
        raise ValueError('Fresh compact extraction coverage differs')
    roles={r['entity_id']:r['role'] for r in read_tsv(ROLES/'owner_roles.tsv')}
    sources=[(FRESH,fresh,'train5k.pkl','train5k_p200k.npy','train5k_owners.npz')]
    for k in range(8):
        item=cloud['partitions'][str(k)];path=Path(item['directory']);done=read(path/'completion.json')
        if digest(path/'completion.json')!=item['completion_sha256']:raise ValueError('Compact receipt changed')
        sources.append((path,done,'train.pkl','p200.npy','owners.npz'))
    rows=[];p200=[];owners=[];truth=[];countries=[]
    for path,done,raw,p2,owner_file in sources:
        for name,h in done['output_sha256'].items():
            if digest(path/name)!=h:raise ValueError('Fresh extraction file changed')
        current=pickle.load((path/raw).open('rb'));values=np.load(path/p2,allow_pickle=False)
        with np.load(path/owner_file,allow_pickle=False) as z:
            oo=list(map(str,z['reference_ids']));owners.extend(oo);truth.extend(z['true_counts']);countries.extend(z['countries'])
        owner_set=set(oo)
        if (len(values)!=len(current) or not np.isfinite(values).all() or ((values<0)|(values>1)).any()
                or any(r[0] not in owner_set or r[4]<.01 for r in current)):
            raise ValueError('Routed compact alignment differs')
        rows.extend(current);p200.extend(values)
    if len(owners)!=30000 or len(set(owners))!=30000 or set(owners)!=set(roles):raise ValueError('Fresh owner roles/coverage mismatch')
    owner_fit,owner_cal=role_masks(owners,roles);pair_fit,pair_cal=role_masks([r[0] for r in rows],roles)
    if owner_fit.sum()!=25000 or owner_cal.sum()!=5000:raise ValueError('Frozen fitting/calibration sizes changed')
    if len({r[:2] for r in rows})!=len(rows):raise ValueError('Duplicate routed fresh pair')
    raw=WORK/'rows.pkl'
    if raw.exists():
        if pickle.load(raw.open('rb'))!=rows:raise ValueError('Retained fresh row assembly differs')
    else:
        with raw.with_suffix('.tmp.pkl').open('wb') as handle:pickle.dump(rows,handle)
        raw.with_suffix('.tmp.pkl').replace(raw)
    np.save(WORK/'p200.npy',np.asarray(p200));np.savez_compressed(WORK/'roles.npz',pair_fit=pair_fit,pair_cal=pair_cal,
        owner_fit=owner_fit,owner_cal=owner_cal,reference_ids=np.asarray(owners),true_counts=np.asarray(truth),countries=np.asarray(countries))
    counts_done=read(COUNTS/'completion.json')
    if not counts_done['complete'] or counts_done['references']!=2206821 or digest(COUNTS/'counts.sqlite3')!=counts_done['database_sha256']:
        raise ValueError('Full training-S1 text counts invalid')
    if not (WORK/'cpu_features.json').exists():
        progress('computing_edit_and_structural_features');counts=Counts(COUNTS/'counts.sqlite3')
        fragments=read(STRUCT/'feature_pilot_v2/fragments.json')['fragments'];ops=[];extras=[]
        for start in range(0,len(rows),10000):
            resources();part=rows[start:start+10000];ops.append(edit_matrix(part,functions));extras.append(structural_matrix(part,counts,fragments))
        counts.close();np.save(WORK/'ops28.npy',np.vstack(ops));np.save(WORK/'structural26.npy',np.vstack(extras))
        # Recount only relevant names, over all provided S1 and target text. No labels.
        spec=importlib.util.spec_from_file_location('frozen_ascii_count_core',core_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        core=lru_cache(maxsize=50000)(module.core);left=[(r[3],core(r[5])) for r in rows];right=[(r[3],core(r[7])) for r in rows]
        interested=set(left)|set(right);values={};db=connect(ROOT/'artifacts/train_store_v2');db.execute('PRAGMA cache_size=-8192')
        for table,expected in [('anchors',2206821),('records',10320219)]:
            counter=Counter();scanned=0
            for scanned,r in enumerate(db.execute('SELECT name,country FROM '+table),1):
                key=(r['country'],core(r['name']))
                if key in interested:counter[key]+=1
                if scanned%500000==0:resources();progress('counting_provided_text_only',table=table,scanned=scanned)
            if scanned!=expected:raise ValueError('Provided text count coverage differs')
            values[table]=counter
        db.close();freq=np.asarray([[np.log1p(values['anchors'][l]),np.log1p(values['anchors'][r]),np.log1p(values['records'][r])]
                                   for l,r in zip(left,right)],np.float32);np.save(WORK/'name_freq3.npy',freq)
        old=len(pickle.load((FRESH/'train5k.pkl').open('rb')))
        if not np.array_equal(freq[:old,2:3],np.load(ROOT/'artifacts/improvement_v3/target_frequency_peerhalf_v1/train5k.npy')):
            raise ValueError('Original fresh5k target-frequency replay differs')
        write(WORK/'cpu_features.json',dict(complete=True,rows_sha256=digest(raw),outputs_sha256={n:digest(WORK/n) for n in ('ops28.npy','structural26.npy','name_freq3.npy')},labels_used=False))
    cpu=read(WORK/'cpu_features.json')
    if cpu['rows_sha256']!=digest(raw):raise ValueError('Cached CPU feature row identity differs')
    for name,h in cpu['outputs_sha256'].items():
        if digest(WORK/name)!=h:raise ValueError('Cached CPU feature changed')
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    for name,weights in checkpoints.items():
        out=WORK/(name+'_pce.npy');receipt=WORK/(name+'_scores.json')
        if receipt.exists():
            done=read(receipt)
            if done['row_sha256']!=digest(raw) or done['scores_sha256']!=digest(out):raise ValueError('Cached fresh CE changed')
            continue
        lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
        while True:
            if time.monotonic()>=end:raise TimeoutError('Finite fresh scoring window expired')
            if not gpu_processes():
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:pass
            progress('waiting_for_gpu',checkpoint=name);time.sleep(30)
        try:
            resources(min_ram_gib=4);progress('gpu_scoring_fresh_rows',checkpoint=name)
            if digest(weights)!=identity['checkpoint_sha256'][name]:raise ValueError('Frozen checkpoint changed')
            with (WORK/'gpu.log').open('a') as log:
                subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),'tools/v3_ce_half_score.py','--model-dir',
                    'artifacts/improvement_v3/transfer_pilot_v1/models/e5','--weights',str(weights),'--input',str(raw),'--output',str(out)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(1800,end-time.monotonic()))
        finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
        if digest(weights)!=identity['checkpoint_sha256'][name]:raise ValueError('Checkpoint changed during GPU scoring')
        p=np.load(out,allow_pickle=False)
        if p.shape!=(len(rows),) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid fresh CE output')
        write(receipt,dict(complete=True,row_sha256=digest(raw),checkpoint_sha256=digest(weights),scores_sha256=digest(out),precision='FP16 weights, batch512'))
    if digest(core_path)!=identity['core_sha256'] or any(digest(p)!=h for p,h in functions[2].items()):raise ValueError('Feature definition changed during job')
    names=['rows.pkl','roles.npz','p200.npy','ops28.npy','structural26.npy','name_freq3.npy','parent_pce.npy','round3_pce.npy']
    write(WORK/'completion.json',dict(complete=True,owners=30000,fit_owners=25000,calibration_owners=5000,
        routed_pairs=len(rows),role_manifest_sha256=identity['role_manifest_sha256'],identity_sha256=digest(WORK/'identity.json'),
        outputs_sha256={n:digest(WORK/n) for n in names},model_fitted=False,calibration_metrics_inspected=False,
        ready_for_submission=False,note='Features only. Enforce role masks; fresh calibration needs honest rival coverage before policy selection.'))
    progress('complete_features_only')


if __name__=='__main__':run()
