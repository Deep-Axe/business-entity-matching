"""Bounded follow-on: score new stacker training rows after the queued GPU pass.

Requires complete fresh training data and a free cooperative GPU lease. Produces
an evaluation, not an automatically promoted or uploaded submission.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from v3_neural_test_pipeline import gpu_processes,resources,SCRATCH
from v3_transfer_data import ROOT,digest,write


def run(minutes=540,tag='e5_pilot400k',work=None,neural_output=None,rival_scores=None,ce_root=None,gpu_worker=None):
    work=Path(work or ROOT/'artifacts/improvement_v3/fresh_stack_followup_v1');work.mkdir(parents=True,exist_ok=True)
    data=ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'
    neural_output=Path(neural_output or ROOT/'artifacts/duf-fel_full_output_v4_neural400k_ops_v2')
    gpu_worker=Path(gpu_worker or ROOT/'tools/v3_ce_score_partition.py')
    lock=(work/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60
    def progress(stage,**kw):
        write(work/'progress.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kw));print(stage,flush=True)
    lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
    while time.monotonic()<end:
        # Do not jump ahead of the already queued test pass. It explicitly
        # releases the GPU lease before its CPU assembly/validation phase.
        neural=neural_output/'progress.json'
        phase=json.loads(neural.read_text()).get('stage') if neural.exists() else None
        ready=phase in ('owner_assignment_and_assembly','stream_validation','official_validation','complete')
        # The retained-peer pipeline finishes all CUDA validation before a
        # separate CPU-only rescore. Its explicit parity/audit receipts make
        # this phase safe for the tiny follow-up; older interleaved pipelines
        # must still wait for assembly. This changes scheduling, not scoring.
        if phase=='cpu_rescoring':
            parity=neural_output/'half_calibration/parity.json'
            audit=neural_output/'evaluation/report.json'
            ready=(parity.exists() and audit.exists()
                   and json.loads(parity.read_text()).get('complete',False)
                   and json.loads(audit.read_text()).get('complete',False))
        if ready and (data/'completion.json').exists() and not gpu_processes():
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        progress('waiting_for_fresh_data_and_completed_test_gpu_phase');time.sleep(30)
    else:raise TimeoutError('Fresh stack follow-up finite window expired')
    resources();weights=SCRATCH/tag/'model.pt';model_hash=digest(weights)
    source=data/'train5k.pkl';input_hash=digest(source);dest=data/f'train5k.{tag}.npy'
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);progress('gpu_fresh_training_rows')
    with (work/'gpu.log').open('a') as log:
        subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),str(gpu_worker),
            '--model-dir','artifacts/improvement_v3/transfer_pilot_v1/models/e5','--weights',str(weights),
            '--input',str(source),'--output',str(dest)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=end-time.monotonic())
    fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    if digest(weights)!=model_hash or digest(source)!=input_hash:raise ValueError('Fresh scoring input changed')
    write(data/f'train5k.{tag}.receipt.json',dict(complete=True,checkpoint_sha256=model_hash,
        input_sha256=input_hash,probabilities_sha256=digest(dest),gpu_worker_sha256=digest(gpu_worker)))
    progress('cpu_fit_and_audit')
    extra=['--rival-scores',str(rival_scores)] if rival_scores else []
    if ce_root:extra+=['--ce-root',str(ce_root)]
    with (work/'audit.log').open('a') as log:
        subprocess.run([sys.executable,'tools/v3_neural_independent_audit.py','--tag',tag,'--training-root',str(data),
            '--output',str(work/'evaluation'),'--ablations',*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=end-time.monotonic())
    report=json.loads((work/'evaluation/report.json').read_text())
    progress('complete',selected=report['selected'],development=report['models'][report['selected']['name']]['development'],
             ready_for_submission=False,note='Evaluate diagnostics before any separate CPU-only rescore of retained CE test predictions')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--minutes',type=float,default=540)
    p.add_argument('--tag',default='e5_pilot400k');p.add_argument('--work',type=Path)
    p.add_argument('--neural-output',type=Path);p.add_argument('--rival-scores',type=Path)
    p.add_argument('--ce-root',type=Path);p.add_argument('--gpu-worker',type=Path)
    run(**vars(p.parse_args()))
