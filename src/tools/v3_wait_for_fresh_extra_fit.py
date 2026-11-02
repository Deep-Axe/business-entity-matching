"""Finite CPU-only follow-up once the precision-matched fresh CE rows exist."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from v3_transfer_data import ROOT,write


def run():
    work=ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_extra_v1';work.mkdir(parents=True,exist_ok=True)
    prerequisite=ROOT/'artifacts/improvement_v3/fresh_stack_peerhalf_v1/evaluation/report.json'
    peer=ROOT/'artifacts/duf-fel_full_output_v4_neural1300k_peerhalf_v1'
    extra=ROOT/'artifacts/improvement_v3/structural_neural_extra1300k_v1'
    end=time.monotonic()+12*3600
    while not (prerequisite.exists() and (extra/'manifest.json').exists()):
        if time.monotonic()>=end:raise TimeoutError('Finite fresh-extra waiting window expired')
        write(work/'progress.json',dict(stage='waiting_for_precision_matched_fresh_training_rows',ready_for_submission=False))
        time.sleep(30)
    if not json.loads(prerequisite.read_text())['complete']:raise ValueError('Prerequisite audit incomplete')
    write(work/'progress.json',dict(stage='cpu_extra_feature_fit',ready_for_submission=False))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('PYTHONPATH',None)
    with (work/'audit.log').open('a') as log:
        subprocess.run([sys.executable,'tools/v3_neural_independent_audit.py','--tag','e5_cont1300k',
            '--training-root',str(ROOT/'artifacts/improvement_v3/fresh_stack_train_v1'),
            '--ce-root',str(peer/'half_calibration'),'--rival-scores',str(peer/'half_calibration/rivals.npy'),
            '--extra-root',str(extra),'--output',str(work/'evaluation')],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=end-time.monotonic(),check=True)
    report=json.loads((work/'evaluation/report.json').read_text())
    write(work/'progress.json',dict(stage='complete',selected=report['selected'],
        development=report['models'][report['selected']['name']]['development'],ready_for_submission=False,
        note='Evaluation only; test probabilities can be rescored on CPU after model/feature review'))


if __name__=='__main__':run()
