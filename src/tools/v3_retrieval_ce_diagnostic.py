"""After submission-critical GPU work, test CE on honest extra candidates.

Reports pair counts only, not a full-pipeline gain. No invented p100 score,
first-stage injection, submission edits, or extra retrieval. Whole-owner/global
competition validation would still be needed before any deployment.
"""
import argparse
import fcntl
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from v3_neural_test_pipeline import read,resources,gpu_processes
from v3_peer_neural_submission import valid_probabilities
from v3_transfer_data import ROOT,digest,write


def counts(y,p,threshold):
    keep=p>=threshold;tp=int((keep&(y==1)).sum());fp=int((keep&(y==0)).sum())
    return dict(threshold=float(threshold),tp=tp,fp=fp,fn=int((y==1).sum())-tp,
        pair_precision=tp/(tp+fp) if tp+fp else None)


def run(source=None,output=None):
    source=Path(source or ROOT/'artifacts/improvement_v3/retrieval_error_slices_v1')
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    fresh=ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_v1/evaluation/report.json'
    wider=ROOT/'artifacts/improvement_v3/widened_rival_neural_v1/evaluation/report.json'
    output=Path(output or ROOT/'artifacts/improvement_v3/retrieval_ce_diagnostic_v1');output.mkdir(parents=True,exist_ok=True)
    end=time.monotonic()+12*3600;lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
    while True:
        if time.monotonic()>=end:raise TimeoutError('Finite retrieval CE diagnostic window expired')
        if all(p.exists() for p in (source/'report.json',peer/'completion.json',fresh,wider)) and not gpu_processes():
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        write(output/'progress.json',dict(stage='waiting_for_submission_and_critical_audits',ready_for_submission=False));time.sleep(30)
    resources();report=read(source/'report.json');parity=read(peer/'half_calibration/parity.json');weights=peer/'reproduction/model.pt'
    if not report['complete'] or not parity['complete'] or parity['checkpoint_sha256']!=digest(weights):
        raise ValueError('Missing raw retrieval or verified checkpoint identity')
    identity=dict(source_report_sha256=digest(source/'report.json'),checkpoint_sha256=digest(weights),
        worker_sha256=digest(ROOT/'tools/v3_ce_half_score.py'),script_sha256=digest(__file__))
    if (output/'identity.json').exists():
        if read(output/'identity.json')!=identity:raise ValueError('Retrieval diagnostic identity changed')
    else:write(output/'identity.json',identity)
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    split_data={}
    try:
        for split in ('tune','dev'):
            raw=source/f'{split}_extra.pkl';dest=output/f'{split}_pce.npy';receipt=output/f'{split}_scores.json'
            if digest(raw)!=report['outputs_sha256'][raw.name]:raise ValueError('Extra-candidate text changed')
            write(output/'progress.json',dict(stage='gpu_extra_candidate_diagnostic',split=split,ready_for_submission=False))
            if receipt.exists():
                done=read(receipt)
                if done['identity_sha256']!=digest(output/'identity.json') or done['scores_sha256']!=digest(dest):
                    raise ValueError('Retained diagnostic scores changed')
            else:
                with (output/'gpu.log').open('a') as log:
                    subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),'tools/v3_ce_half_score.py',
                        '--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5',
                        '--weights',str(weights),'--input',str(raw),'--output',str(dest)],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
                write(receipt,dict(complete=True,identity_sha256=digest(output/'identity.json'),scores_sha256=digest(dest)))
            rows=pickle.load(raw.open('rb'));p=np.load(dest,allow_pickle=False)
            if not valid_probabilities(p,len(rows)) or any(r[4] is not None for r in rows):
                raise ValueError('Invalid extra-candidate probabilities or invented p100k')
            split_data[split]=(np.asarray([r[2] for r in rows]),p)
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    thresholds=(.90,.95,.98,.99,.995,.999,.9999,1.000001)
    grid={s:[counts(y,p,t) for t in thresholds] for s,(y,p) in split_data.items()}
    # Fixed exploratory screen, not a deployment threshold: highest recall with
    # zero false additions in the small tune cohort. Reject-all is available.
    choice=max((r for r in grid['tune'] if r['fp']==0),key=lambda r:(r['tp'],r['threshold']))
    dev=counts(*split_data['dev'],choice['threshold'])
    write(output/'report.json',dict(complete=True,grid=grid,tune_only_screen=choice,development_at_tune_screen=dev,
        identity_sha256=digest(output/'identity.json'),ready_for_submission=False,full_pipeline_gain_evaluated=False,
        note='Extremely low new-positive prevalence; zero tune FPs is not a generalization guarantee. New-target rival coverage and full-owner macro F0.5 remain untested. No automatic candidate additions.'))
    write(output/'progress.json',dict(stage='complete',ready_for_submission=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path)
    run(**vars(p.parse_args()))
