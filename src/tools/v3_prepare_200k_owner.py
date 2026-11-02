"""Finite, resumable 200k owner evaluation and separately validated submission.

Never alters the raw 200k release, 100k submission, GPU job, or AWS resources.
Run inside the shared amazon-ml.slice; use one CPU and a bounded memory cgroup.
No competition uploads. A worse development result is not auto-promoted.
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

from v3_transfer_data import ROOT,digest,write

WORK=ROOT/'artifacts/improvement_v3/owner200k_v1'
EVAL=WORK/'evaluation'
CACHE=WORK/'test'
MODEL=ROOT/'artifacts/improvement_v2/cpu/fit_gate0_200k_v1/enhanced.cbm'
TEXT=ROOT/'artifacts/improvement_v3/structural_v1/text_scorer.json'
CONTROLLER=ROOT/'artifacts/improvement_v2/inference/submission_controller_200k_v1'
BASELINE=ROOT/'artifacts/duf-fel_full_output_v2_200k'
OUTPUT=ROOT/'artifacts/duf-fel_full_output_v3_200k_owner_v1'
DATA=ROOT/'6ab10eb3b23ba_student_resource/student_resource/dataset/test'


def read(path):return json.loads(Path(path).read_text())


def promotion(report,policy):
    dev=report['splits']['dev']
    return (policy.get('enabled',False) and
            dev['macro_f05']>max(dev['baseline_macro_f05'],.971091326095)+1e-12)


def run(minutes=440,interval=45):
    WORK.mkdir(parents=True,exist_ok=True)
    lock=(WORK/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();end=start+minutes*60
    model_report=read(MODEL.parent/'enhanced_report.json')
    threshold=model_report['threshold']
    identity=dict(model_sha256=digest(MODEL),threshold=threshold,text_sha256=digest(TEXT),
        code_sha256={name:digest(ROOT/'tools'/name) for name in
            ('v3_owner_assignment.py','v3_owner_evaluate.py','v3_prepare_200k_owner.py',
             'v3_validate_owner_submission.py')})
    if (WORK/'identity.json').exists():
        if read(WORK/'identity.json')!=identity:raise ValueError('Owner workflow identity changed')
    else:write(WORK/'identity.json',identity)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    def update(stage,**kw):
        result=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),
                    ready_for_submission=False,uploaded_to_portal=False)
        result.update(kw)
        write(WORK/'progress.json',result);print(json.dumps(result),flush=True)
    def execute(stage,args):
        remaining=end-time.monotonic()
        if remaining<=0:raise TimeoutError('Finite owner workflow window expired')
        if shutil.disk_usage(ROOT).free<6*1024**3:
            raise RuntimeError('Below 6GiB free: stop this experiment to protect baseline')
        update(stage)
        with (WORK/(stage+'.log')).open('a') as log:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    def evaluation(command,*extra):
        execute(command,['tools/v3_owner_evaluate.py',command,'--output',EVAL,*extra])
    try:
        if not (EVAL/'policy.json').exists():
            evaluation('setup','--model-size','200000')
            evaluation('reuse','--previous',ROOT/'artifacts/improvement_v3/structural_v1/owner_eval')
            evaluation('cached-profiles')
            evaluation('discover')
            evaluation('shortlist')
            evaluation('score')
            evaluation('evaluate')
        policy=read(EVAL/'policy.json');report=read(EVAL/'report.json')
        if not promotion(report,policy):
            update('not_promoted_development_gate',report=str(EVAL/'report.json'),
                   note='Raw 200k packaging continues independently; no inferior variant auto-promoted')
            return
        while time.monotonic()<end:
            execute('ingest',['tools/v3_owner_assignment.py','ingest','--controller',CONTROLLER,
                             '--output',CACHE,'--model',MODEL,'--threshold',threshold])
            if read(CACHE/'progress.json')['complete_ingestion'] and (BASELINE/'assembly.json').exists():break
            update('waiting_for_complete_200k_baseline',development=report['splits']['dev'])
            time.sleep(min(interval,max(0,end-time.monotonic())))
        else:raise TimeoutError('Baseline not complete within finite owner workflow window')
        if not (OUTPUT/'report.json').exists():
            execute('apply',['tools/v3_owner_assignment.py','apply','--cache',CACHE,
                '--policy',EVAL/'policy.json','--text-scorer',TEXT,'--baseline-output',BASELINE,
                '--output',OUTPUT,'--model',MODEL])
        if not (OUTPUT/'stream_validation.json').exists():
            execute('stream_validate',['tools/validate_full_submission.py','--test-dir',DATA,
                '--output',OUTPUT,'--report',OUTPUT/'stream_validation.json'])
        if not (OUTPUT/'official_matching_validation.json').exists():
            execute('official_validate',['tools/package_v2_200k_submission.py','official-validate',
                '--validator',DATA.parent.parent/'utils/validate_submission.py','--test-dir',DATA,
                '--output',OUTPUT,'--report',OUTPUT/'official_matching_validation.json'])
        if not (OUTPUT/'completion.json').exists():
            execute('structural_validate',['tools/v3_validate_owner_submission.py','--cache',CACHE,
                '--baseline-output',BASELINE,'--output',OUTPUT,'--policy',EVAL/'policy.json',
                '--text-scorer',TEXT])
        receipt=read(OUTPUT/'completion.json')
        if not receipt['ready_for_submission']:raise ValueError('Final receipt not ready')
        if receipt['output_sha256']['matching_results.tsv']!=digest(OUTPUT/'matching_results.tsv'):
            raise ValueError('Final matching identity changed')
        update('complete',**{'ready_for_submission':True},matching=str(OUTPUT/'matching_results.tsv'),
               development=report['splits']['dev'],note='No French labels or measured leaderboard score')
    except Exception as error:
        update('attention_required',error=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes',type=float,default=440);p.add_argument('--interval',type=float,default=45)
    args=p.parse_args()
    if args.minutes<=0 or not 1<=args.interval<=60:p.error('Positive finite duration and 1–60 second polling required')
    run(**vars(args))
