"""Finite lower-priority name-specialist pilot after submission-critical work.

Uses a separate checkpoint and a single guarded GPU lease. No full-test scoring
or automatic promotion. Two fresh-owner CPU fits select on existing calibration,
then report reused development and error slices. Current submissions are intact.
"""
import fcntl
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from v3_neural_test_pipeline import SCRATCH,read,resources,gpu_processes,lg
from v3_transfer_data import ROOT,digest,write


def missing(value):return not value.strip() or value.strip().upper() in ('NULL','<NULL>')


def matrix(rows,scores):
    mask=np.asarray([missing(r[8]) for r in rows])
    if scores.shape!=(len(rows),) or not np.isnan(scores[~mask]).all() or not np.isfinite(scores[mask]).all():
        raise ValueError('Specialist route alignment or values invalid')
    if ((scores[mask]<0)|(scores[mask]>1)).any():raise ValueError('Invalid specialist probabilities')
    values=np.zeros((len(rows),1),np.float32);values[mask,0]=lg(scores[mask])
    return values


def run():
    work=ROOT/'artifacts/improvement_v3/empty_name_specialist_v1';work.mkdir(parents=True,exist_ok=True)
    lock=(work/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    data=ROOT/'artifacts/improvement_v3/empty_name_specialist_data_v1'
    fresh=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'
    baseline=ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_v1/evaluation'
    structural=ROOT/'artifacts/improvement_v3/structural_neural_extra1300k_v1'
    prerequisites=[peer/'completion.json',baseline/'report.json',data/'completion.json']
    for name in ('widened_rival_neural_v1/evaluation','retrieval_ce_diagnostic_v1','native_alias_ce_diagnostic_v1','website_alias_ce_diagnostic_v1'):
        prerequisites.append(ROOT/'artifacts/improvement_v3'/name/'report.json')
    end=time.monotonic()+12*3600;lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a');quiet=None
    def progress(stage,**kw):write(work/'progress.json',dict(stage=stage,ready_for_submission=False,**kw))
    while True:
        if time.monotonic()>=end:raise TimeoutError('Finite name-specialist waiting window expired')
        ready=all(p.exists() for p in prerequisites)
        if ready and read(peer/'completion.json')['ready_for_submission'] and not gpu_processes():
            if quiet is None:quiet=time.monotonic()
            if time.monotonic()-quiet>=60:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:quiet=None
        else:quiet=None
        progress('waiting_for_validated_submission_and_critical_gpu_diagnostics');time.sleep(30)
    resources(min_disk_gib=8)
    identity=dict(data_sha256=digest(data/'completion.json'),parent_checkpoint_sha256=digest(peer/'reproduction/model.pt'),
        gpu_worker_sha256=digest(ROOT/'tools/v3_empty_name_gpu.py'),pipeline_sha256=digest(__file__),
        parent_precision_sha256=digest(peer/'half_calibration/parity.json'),
        route='Additional feature only for naturally missing target addresses; original general encoder retained',
        fitting='Fresh 5k train-fold owners; fixed existing calibration; reused development reported')
    if (work/'identity.json').exists():
        if read(work/'identity.json')!=identity:raise ValueError('Name-specialist pilot identity changed')
    else:write(work/'identity.json',identity)
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    def command(argv,log,limit=3600):
        resources()
        with (work/log).open('a') as handle:
            subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,
                check=True,timeout=min(limit,end-time.monotonic()))
    model=work/'model';score_dir=work/'scores';score_dir.mkdir(exist_ok=True)
    raw={s:SCRATCH/f'{s}.pkl' for s in ('tune','dev','rivals')};raw['train5k']=fresh/'train5k.pkl'
    try:
        if not (model/'completion.json').exists():
            progress('training_name_only_specialist')
            command([ROOT/'.venv-cascade-v2/bin/python','tools/v3_empty_name_gpu.py','train',
                '--data',data,'--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5',
                '--weights',peer/'reproduction/model.pt','--output',model],'training.log',3900)
        done=read(model/'completion.json')
        if not done['complete'] or digest(model/'model.pt')!=done['model_sha256']:raise ValueError('Specialist checkpoint failed validation')
        for split,path in raw.items():
            dest=score_dir/f'{split}.npy';receipt=dest.with_suffix('.receipt.json')
            if receipt.exists():
                old=read(receipt)
                if old['input_sha256']!=digest(path) or old['checkpoint_sha256']!=done['model_sha256'] or old['scores_sha256']!=digest(dest):
                    raise ValueError('Retained specialist scores changed')
                continue
            progress('scoring_empty_address_pilot',split=split)
            command([ROOT/'.venv-cascade-v2/bin/python','tools/v3_empty_name_gpu.py','score',
                '--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5','--weights',model/'model.pt',
                '--input',path,'--output',dest],'scoring.log',600)
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    old_manifest=read(structural/'manifest.json');variants={}
    for combined in (False,True):
        root=work/('combined_features' if combined else 'specialist_features');root.mkdir(exist_ok=True);bound={}
        names=['empty_address_name_specialist_logit']
        if combined:names=old_manifest['names']+names
        for split,path in raw.items():
            rows=pickle.load(path.open('rb'));scores=np.load(score_dir/f'{split}.npy',allow_pickle=False);x=matrix(rows,scores)
            if combined:
                info=old_manifest['inputs'][split];original=structural/f'{split}.npy'
                if info['sha256']!=digest(path) or info['features_sha256']!=digest(original):raise ValueError('Original structural rows changed')
                x=np.column_stack((np.load(original,allow_pickle=False),x))
            dest=root/f'{split}.npy';np.save(dest,x)
            bound[split]=dict(path=str(path),sha256=digest(path),pairs=len(rows),features_sha256=digest(dest))
        write(root/'manifest.json',dict(complete=True,names=names,inputs=bound,pilot_identity_sha256=digest(work/'identity.json'),
            specialist_checkpoint_sha256=done['model_sha256'],score_receipts_sha256={s:digest(score_dir/f'{s}.receipt.json') for s in raw},
            original_structural_manifest_sha256=digest(structural/'manifest.json') if combined else None,
            note='Additional conditional feature, not replacement of the parent neural score. Zero outside empty-address route.'))
        audit=root/'evaluation'
        if not (audit/'report.json').exists():
            progress('cpu_fresh_owner_fit',combined_structural=combined)
            command([sys.executable,'tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
                '--ce-root',peer/'half_calibration','--rival-scores',peer/'half_calibration/rivals.npy',
                '--training-root',fresh,'--extra-root',root,'--output',audit],root.name+'.log',1200)
        report=read(audit/'report.json');variants[root.name]=dict(report_sha256=digest(audit/'report.json'),selected=report['selected'],
            development=report['models'][report['selected']['name']]['development'])
    write(work/'report.json',dict(complete=True,variants=variants,baseline_report_sha256=digest(baseline/'report.json'),
        ready_for_submission=False,full_test_scored=False,
        note='Pilot only. Inspect calibration, wider rivals, slice/country tradeoffs and resource headroom before any full-test route.'))
    progress('complete',variants=variants)


if __name__=='__main__':run()
