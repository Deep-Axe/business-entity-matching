"""Wait for the existing larger CE training, then audit it without overlapping GPU work."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from v3_neural_test_pipeline import gpu_processes,resources,SCRATCH
from v3_transfer_data import ROOT,digest,write


def run(tag,output,minutes=150):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60;training_log=SCRATCH/(tag+'.log')
    gpu_lock=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
    def progress(stage,**kw):
        write(output/'progress.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kw))
        print(stage,flush=True)
    while time.monotonic()<end:
        finished=training_log.exists() and any(line.startswith('done ') for line in training_log.read_text().splitlines())
        if finished and not gpu_processes():
            try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        progress('waiting_for_finished_training_and_gpu_slot');time.sleep(10)
    else:raise TimeoutError('Finite follow-up audit window expired')
    resources();weights=SCRATCH/tag/'model.pt'
    retained=output/'model.pt'
    if not retained.exists():os.link(weights,retained)
    if digest(retained)!=digest(weights):raise ValueError('Retained follow-up checkpoint changed')
    inputs={str(p):digest(p) for p in (weights,SCRATCH/tag/'tune_pce.npy',SCRATCH/tag/'dev_pce.npy')}
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    for split in ('tune','rivals'):
        progress('gpu_'+split)
        with (output/'gpu.log').open('a') as log:
            subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),'tools/v3_ce_score_partition.py',
                '--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5','--weights',str(retained),
                '--input',str(SCRATCH/f'{split}.pkl'),'--output',str(output/f'{split}.npy')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
        if split=='tune':
            delta=float(np.max(np.abs(np.load(output/'tune.npy')-np.load(SCRATCH/tag/'tune_pce.npy'))))
            if delta>2e-4:raise ValueError('New checkpoint failed saved tuning prediction replay')
    fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    for path,value in inputs.items():
        if digest(path)!=value:raise ValueError('Training artifacts changed during follow-up')
    write(output/'checkpoint_receipt.json',dict(complete=True,inputs_sha256=inputs,
        tune_replay_max_absolute_delta=delta,rival_scores_sha256=digest(output/'rivals.npy')))
    progress('cpu_audit')
    with (output/'audit.log').open('a') as log:
        subprocess.run([sys.executable,'tools/v3_neural_independent_audit.py','--tag',tag,
            '--rival-scores',str(output/'rivals.npy'),'--output',str(output/'evaluation'),'--ablations'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
    report=json.loads((output/'evaluation/report.json').read_text())
    progress('complete',selected=report['selected'],development=report['models'][report['selected']['name']]['development'],
             submission_ready=False,note='Evaluation only; does not silently replace the queued 400k submission')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=150);a=p.parse_args()
    try:run(**vars(a))
    except Exception as e:
        write(a.output/'failure.json',dict(error=str(e),ready_for_submission=False));raise
