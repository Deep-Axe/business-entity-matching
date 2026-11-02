"""Finite low-priority completion of the isolated structural pilots.

No cloud actions, submission upload, production config edits or GPU work.
Resumes committed rival profiles and trial reports. A pilot error stops this
queue only. Final TSV packaging is deliberately left for reviewed handoff.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from v3_transfer_data import ROOT,write,digest

BASE=ROOT/'artifacts/improvement_v3/structural_v1'


def run(hours=8):
    logs=BASE/'completion_v2';logs.mkdir(parents=True,exist_ok=True)
    lock=(logs/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Prefer killing an experimental subprocess to killing production inference
    # if some other application exhausts system memory.
    try:Path('/proc/self/oom_score_adj').write_text('500')
    except OSError:pass
    start=time.monotonic();deadline=start+hours*3600
    scripts=['v3_owner_assignment.py','v3_owner_evaluate.py','v3_structural_features.py','v3_structural_pilot.py']
    hashes={s:digest(ROOT/'tools'/s) for s in scripts}
    receipt=logs/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text())!=hashes:raise ValueError('Pilot code changed after queue launch')
    write(receipt,hashes)
    def update(stage,**extra):
        value={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'complete':False,**extra}
        write(logs/'progress.json',value);print(json.dumps(value),flush=True)
    def safety():
        if time.monotonic()>deadline:raise TimeoutError('Finite pilot time budget reached; committed work remains resumable')
        if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Pilot stopped to preserve8GiB disk headroom')
        values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(values['MemAvailable'].split()[0])<2*1024**2:raise RuntimeError('Pilot stopped: system available RAM below2GiB')
    def execute(stage,args,completion):
        if completion.exists():return
        safety()
        if any(digest(ROOT/'tools'/s)!=h for s,h in hashes.items()):raise ValueError('Pilot code changed')
        update(stage)
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        with (logs/f'{stage}.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:safety();time.sleep(2)
                if proc.returncode:raise RuntimeError(f'{stage} exited{proc.returncode}; inspect its log')
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise
    try:
        while not ((BASE/'owner_eval/shortlist.json').exists() and (BASE/'feature_pilot_v2/dev.complete.json').exists()):
            safety();update('waiting_for_rival_shortlist_and_feature_preparation');time.sleep(20)
        # Scoring completion is explicitly checked by evaluate; no existence-only
        # progress receipt can authorize incomplete evaluation.
        execute('rival_score',['tools/v3_owner_evaluate.py','score'],BASE/'owner_eval/policy.json')
        execute('owner_evaluate',['tools/v3_owner_evaluate.py','evaluate'],BASE/'owner_eval/policy.json')
        execute('stacker_fit',['tools/v3_structural_pilot.py','fit','--with-rivals'],BASE/'feature_pilot_v2/report.json')
        execute('claims_ingest',['tools/v3_owner_assignment.py','ingest'],logs/'never_skip_ingest')
        update('pilots_complete_ready_for_review',complete=True,
               owner_report=str(BASE/'owner_eval/report.json'),stacker_report=str(BASE/'feature_pilot_v2/report.json'),
               submission_ready=False,next='Review gains and limitations, then apply frozen policy only to full100k output and validate/package separately')
    except BaseException as exc:
        update('stopped_safely',error=f'{type(exc).__name__}: {exc}',resumable=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hours',type=float,default=8)
    a=p.parse_args()
    if not 0<a.hours<=12:raise ValueError('Finite pilot duration must be within12hours')
    run(a.hours)
