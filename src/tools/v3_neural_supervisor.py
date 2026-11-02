"""Finite auto-resume for cooperative neural deployment GPU-contention exits.

Never retries integrity/model/coverage errors and never kills foreign jobs.
Completed partitions remain untouched. The fixed wall-clock budget covers every
attempt, including waiting; no unlimited restart loop.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from v3_neural_test_pipeline import gpu_processes,resources
from v3_transfer_data import ROOT,write


def retryable(log_text):
    return ('RuntimeError: Another GPU job is active; yielding without touching it' in log_text
            or 'RuntimeError: Insufficient free GPU memory' in log_text)


def run(audit,output,minutes=420,max_attempts=20):
    audit,output=Path(audit),Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60;quiet=None;attempt=0
    def progress(stage,**kw):
        write(output/'supervisor.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),
            attempt=attempt,max_attempts=max_attempts,remaining_seconds=max(0,end-time.monotonic()),**kw))
        print(stage,flush=True)
    while time.monotonic()<end and attempt<max_attempts:
        resources()
        if (output/'completion.json').exists():
            c=json.loads((output/'completion.json').read_text())
            if c.get('ready_for_submission') is True:
                progress('completed_submission_pipeline');return
        if gpu_processes():quiet=None
        elif quiet is None:quiet=time.monotonic()
        if quiet is None or time.monotonic()-quiet<60:
            progress('waiting_for_idle_gpu');time.sleep(15);continue
        attempt+=1;progress('deployment_attempt')
        gpu_log=output/'gpu_scoring.log';offset=gpu_log.stat().st_size if gpu_log.exists() else 0
        remaining=end-time.monotonic()
        with (output/'supervisor_pipeline.log').open('a') as log:
            completed=subprocess.run([sys.executable,'tools/v3_neural_test_pipeline.py',
                '--audit',str(audit),'--output',str(output),'--minutes',str(remaining/60),
                '--idle-seconds','30'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
        if completed.returncode==0:progress('completed_submission_pipeline');return
        with gpu_log.open() as log:
            log.seek(offset);new_log=log.read()
        if not retryable(new_log):
            progress('attention_required_nonretryable',returncode=completed.returncode)
            raise RuntimeError('Non-contention failure; inspect preserved logs, no blind retry')
        progress('yielded_to_another_gpu_job',returncode=completed.returncode);quiet=None
    progress('finite_resume_window_exhausted')
    raise TimeoutError('Finite GPU resume window/attempt limit exhausted')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=420);p.add_argument('--max-attempts',type=int,default=20)
    a=p.parse_args()
    if a.minutes<=0 or not 1<=a.max_attempts<=20:p.error('Finite positive duration; 1–20 attempts required')
    run(**vars(a))
