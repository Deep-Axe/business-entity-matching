"""Wait for widened empty-target profiles, score them on a free GPU, then audit.

Original competition evidence is preserved in the union. New rivals use the
same FP16-weight checkpoint as the peer-based test pass. No submission changes.
"""
import fcntl
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import os

import numpy as np

from v3_neural_test_pipeline import SCRATCH,gpu_processes,resources,edit_matrix,feature_functions
from v3_transfer_data import ROOT,digest,write


def run():
    work=ROOT/'artifacts/improvement_v3/widened_rival_neural_v1';work.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/improvement_v3/empty_target_rivals_v1'
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    end=time.monotonic()+12*3600;lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
    def progress(stage,**kw):write(work/'progress.json',dict(stage=stage,ready_for_submission=False,**kw))
    while True:
        if time.monotonic()>=end:raise TimeoutError('Finite widened-rival window expired')
        if (source/'completion.json').exists() and (peer/'half_calibration/parity.json').exists() and not gpu_processes():
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        progress('waiting_for_profiles_and_verified_half_checkpoint');time.sleep(30)
    resources();done=json.loads((source/'completion.json').read_text());input=source/'extra_rivals.pkl'
    if not done['complete'] or digest(input)!=done['rows_sha256']:raise ValueError('Widened profile data changed')
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
    try:
        progress('gpu_extra_rivals')
        with (work/'gpu.log').open('a') as log:
            subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),'tools/v3_ce_half_score.py',
                '--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5',
                '--weights',str(peer/'reproduction/model.pt'),'--input',str(input),'--output',str(work/'extra_pce.npy')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    old=pickle.load((SCRATCH/'rivals.pkl').open('rb'));old_ce=np.load(peer/'half_calibration/rivals.npy')
    extra=pickle.load(input.open('rb'));extra_ce=np.load(work/'extra_pce.npy')
    if len(old)!=len(old_ce) or len(extra)!=len(extra_ce):raise ValueError('Widened CE alignment differs')
    rows=list(old);probs=list(map(float,old_ce));keys={r[:2]:i for i,r in enumerate(old)};overlap=0
    if len(keys)!=len(old):raise ValueError('Old rival keys are duplicated')
    for row,p in zip(extra,extra_ce):
        key=row[:2]
        if key in keys:
            before=rows[keys[key]]
            if before[3]!=row[3] or before[5:9]!=row[5:9] or abs(before[4]-row[4])>1e-8:
                raise ValueError('Recomputed rival context/text disagrees with original evidence')
            overlap+=1;continue
        keys[key]=len(rows);rows.append(row);probs.append(float(p))
    with (work/'rivals.pkl').open('wb') as handle:pickle.dump(rows,handle)
    np.save(work/'rivals_pce.npy',np.asarray(probs,np.float32))
    np.save(work/'rivals_ops.npy',edit_matrix(rows,feature_functions()))
    write(work/'union.json',dict(complete=True,old_pairs=len(old),extra_pairs=len(extra),overlap=overlap,union_pairs=len(rows),
        extra_source_sha256=digest(input),old_source_sha256=digest(SCRATCH/'rivals.pkl'),
        outputs_sha256={n:digest(work/n) for n in ('rivals.pkl','rivals_pce.npy','rivals_ops.npy')},
        note='Auxiliary labels are not used. Keep only p100k>=.01 auxiliary claims, matching actual test routing. Baseline replay retains original old-cache competition.'))
    progress('cpu_widened_owner_audit')
    with (work/'audit.log').open('a') as log:
        subprocess.run([sys.executable,'tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
            '--ce-root',str(peer/'half_calibration'),'--rival-rows',str(work/'rivals.pkl'),
            '--rival-scores',str(work/'rivals_pce.npy'),'--rival-ops',str(work/'rivals_ops.npy'),
            '--output',str(work/'evaluation')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
            timeout=end-time.monotonic())
    report=json.loads((work/'evaluation/report.json').read_text())
    progress('complete',selected=report['selected'],development=report['models'][report['selected']['name']]['development'])


if __name__=='__main__':run()
