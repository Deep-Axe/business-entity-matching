"""Reuse a live peer's full-test CE scores only after provenance/precision checks.

Does not control the peer process or modify its files. Retains every completed
score, checks the full original candidate route, waits for a free GPU, reproduces
the peer's FP16-weight recipe, calibrates on tune, then assembles/validates a
separate full submission. A finite deadline and shared cgroup bound this job.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np

from v3_neural_test_pipeline import (SCRATCH,CONTROLLER,STRUCT,DATA,read,lg,resources,
    gpu_processes,feature_functions,edit_matrix,partition_inputs,verify_route,assemble)
from v3_owner_assignment import open_claims,load_text
from v3_transfer_data import ROOT,digest,write


def valid_probabilities(p,n):
    return p.shape==(n,) and np.isfinite(p).all() and not ((p<0)|(p>1)).any()


def producer_identity(pid,tag,original_audit):
    proc=Path('/proc')/str(pid);argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
    script=SCRATCH.parent/'ce_score.py';weights=SCRATCH/tag/'model.pt'
    if str(script) not in argv or '--swap' in argv or '--model' in argv or '--bs' in argv or '--maxlen' in argv:
        raise ValueError('Unreviewed producer/scoring recipe')
    if argv[argv.index('--weights')+1]!=str(weights) or argv[argv.index('--tag')+1]!=tag:
        raise ValueError('Wrong live producer checkpoint or tag')
    if argv[-128:]!=[str(SCRATCH/'test'/f'part_{i:03d}.pkl') for i in range(128)]:
        raise ValueError('Live producer is not the full ordered 128-partition pass')
    tail=(proc/'stat').read_text().rsplit(')',1)[1].split();ticks=int(tail[19])
    boot=next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime '))
    start=boot+ticks/os.sysconf('SC_CLK_TCK')
    expected=read(original_audit/'report.json')
    if not expected['complete'] or expected['tag']!=tag:raise ValueError('Wrong original checkpoint audit')
    return dict(pid=pid,start_ticks=ticks,start_epoch=start,argv=argv,tag=tag,
        script_sha256=digest(script),checkpoint_sha256=digest(weights),original_audit_sha256=digest(original_audit/'report.json'),
        precision='FP16 model weights, no autocast, batch512, max_length128, original record order')


def run(producer_pid,output,minutes=480,tag='e5_cont1300k'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60
    original=ROOT/'artifacts/improvement_v3/neural_followup1300k_v1/evaluation'
    identity_path=output/'producer.json'
    if identity_path.exists():identity=read(identity_path)
    else:
        identity=producer_identity(producer_pid,tag,original)
        identity['consumer_sha256']=digest(__file__)
        identity['half_worker_sha256']=digest(ROOT/'tools/v3_ce_half_score.py')
        write(identity_path,identity)
    if identity['tag']!=tag or identity['consumer_sha256']!=digest(__file__):raise ValueError('Consumer identity changed')
    if identity['half_worker_sha256']!=digest(ROOT/'tools/v3_ce_half_score.py'):raise ValueError('FP16 replay worker changed')
    retained=output/'reproduction';retained.mkdir(exist_ok=True);weights=retained/'model.pt'
    if not weights.exists():shutil.copy2(SCRATCH/tag/'model.pt',weights)
    if digest(weights)!=identity['checkpoint_sha256']:raise ValueError('Retained checkpoint changed')
    shutil.copy2(SCRATCH.parent/'ce_score.py',retained/'peer_ce_score.py')
    parts=output/'scores';parts.mkdir(exist_ok=True)
    functions=feature_functions();queue=read(CONTROLLER/'queue.json')
    if set(queue['complete'])!=set(map(str,range(128))) or queue['assigned']:raise ValueError('Full 100k source required')
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    def progress(stage,**kw):
        write(output/'progress.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),
            ready_for_submission=False,uploaded_to_portal=False,**kw));print(stage,kw,flush=True)
    def alive():
        if time.monotonic()>=end:raise TimeoutError('Finite peer reuse window expired')
        resources()
    def command(args,log):
        alive()
        with (output/log).open('a') as handle:
            subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,
                           timeout=end-time.monotonic(),check=True)
    # Preserve/check every routed partition as it arrives. No test inference here.
    for part in range(128):
        alive();source=SCRATCH/'test'/f'part_{part:03d}.pkl';peer=Path(str(source)+f'.{tag}.npy')
        receipt=parts/f'{part:03d}.json';dest=parts/f'{part:03d}.ce.npy'
        if receipt.exists():
            old=read(receipt)
            if digest(source)!=old['input_sha256'] or digest(dest)!=old['ce_sha256']:raise ValueError('Retained peer scores changed')
            continue
        while not peer.exists():
            alive();progress('waiting_for_peer_test_scores',completed_partitions=part,partitions=128);time.sleep(20)
        # The peer skips existing files: refuse pre-run outputs of unknown origin.
        if peer.stat().st_mtime<identity['start_epoch'] or source.stat().st_mtime>identity['start_epoch']:
            raise ValueError('Peer output predates producer or source changed after launch')
        if digest(SCRATCH.parent/'ce_score.py')!=identity['script_sha256']:raise ValueError('Peer scorer source changed')
        expected,owners=partition_inputs(part,queue['complete'][str(part)])
        rows=pickle.load(source.open('rb'));verify_route(rows,expected,owners)
        values=np.load(peer,allow_pickle=False)
        if not valid_probabilities(values,len(rows)):raise ValueError('Invalid peer score array')
        source_hash=digest(source);peer_hash=digest(peer);np.save(dest,values)
        if digest(peer)!=peer_hash or digest(dest)!=peer_hash:raise ValueError('Peer scores changed during retention')
        write(receipt,dict(partition=part,pairs=len(rows),references=len(owners),input_sha256=source_hash,
            ce_sha256=peer_hash,producer_sha256=digest(identity_path),source_pair_route_verified=True))
        progress('retained_peer_test_scores',completed_partitions=part+1,partitions=128)
        del rows,expected,owners,values
    calibration=output/'half_calibration';calibration.mkdir(exist_ok=True)
    if not (calibration/'parity.json').exists():
        # Do not interrupt/overlap any GPU process, including the original peer.
        lease=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a');quiet=None
        while True:
            alive()
            if gpu_processes():quiet=None
            elif quiet is None:quiet=time.monotonic()
            if quiet is not None and time.monotonic()-quiet>=60:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:quiet=None
            progress('waiting_for_half_precision_validation_gpu');time.sleep(15)
        try:
            for split,source,dest in [
                ('test_replay',SCRATCH/'test/part_000.pkl',calibration/'test_replay.npy'),
                ('tune',SCRATCH/'tune.pkl',calibration/'tune_pce.npy'),
                ('dev',SCRATCH/'dev.pkl',calibration/'dev_pce.npy'),
                ('rivals',SCRATCH/'rivals.pkl',calibration/'rivals.npy')]:
                progress('half_precision_gpu_validation',split=split)
                command([ROOT/'.venv-cascade-v2/bin/python','tools/v3_ce_half_score.py',
                    '--model-dir',ROOT/'artifacts/improvement_v3/transfer_pilot_v1/models/e5',
                    '--weights',weights,'--input',source,'--output',dest],'half_gpu.log')
                if split=='test_replay':
                    delta=float(np.max(np.abs(np.load(dest)-np.load(parts/'000.ce.npy'))))
                    if delta>2e-4:raise ValueError('Peer FP16 probabilities failed independent replay')
            if digest(SCRATCH/tag/'model.pt')!=identity['checkpoint_sha256']:raise ValueError('Peer checkpoint changed')
            write(calibration/'parity.json',dict(complete=True,test_replay_max_absolute_delta=delta,
                checkpoint_sha256=digest(weights),half_worker_sha256=digest(ROOT/'tools/v3_ce_half_score.py'),
                scores_sha256={p.name:digest(p) for p in calibration.glob('*.npy')}))
        finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    parity=read(calibration/'parity.json')
    for name,value in parity['scores_sha256'].items():
        if digest(calibration/name)!=value:raise ValueError('Calibration probabilities changed')
    audit=output/'evaluation'
    if not (audit/'report.json').exists():
        progress('half_precision_cpu_audit')
        command([sys.executable,'tools/v3_neural_independent_audit.py','--tag',tag,'--ce-root',calibration,
            '--rival-scores',calibration/'rivals.npy','--output',audit],'audit.log')
    report=read(audit/'report.json');selected=report['selected'];metrics=report['models'][selected['name']]['development']
    original_report=read(original/'report.json');old=original_report['models'][original_report['selected']['name']]['development']
    if metrics['macro_f05']<old['macro_f05']-.001 or metrics['macro_f05']<=report['baseline_replay']['macro_f05']:
        raise ValueError('Peer-precision variant failed quality gate; do not auto-submit')
    model=None if selected['name'].startswith('blend_') else joblib.load(audit/(selected['name']+'.joblib'))
    if model is not None and report['feature_columns'][selected['name']]!=list(range(30)):raise ValueError('Unsupported calibrated feature set')
    db=open_claims(output/'cache')
    for part in range(128):
        alive();source=SCRATCH/'test'/f'part_{part:03d}.pkl';bound=read(parts/f'{part:03d}.json')
        recipe=json.dumps(dict(part=bound,audit_sha256=digest(audit/'report.json')),sort_keys=True)
        old=db.execute('SELECT identity FROM partitions WHERE partition=?',(part,)).fetchone()
        if old:
            if old[0]!=recipe:raise ValueError('Scored partition identity changed')
            continue
        if digest(source)!=bound['input_sha256'] or digest(parts/f'{part:03d}.ce.npy')!=bound['ce_sha256']:raise ValueError('Scoring input changed')
        expected,owners=partition_inputs(part,queue['complete'][str(part)])
        rows=pickle.load(source.open('rb'));verify_route(rows,expected,owners);del expected
        ce=np.load(parts/f'{part:03d}.ce.npy');base=np.asarray([r[4] for r in rows])
        if model is None:
            w=float(selected['name'].split('_')[1]);scores=1/(1+np.exp(-((1-w)*lg(base)+w*lg(ce))))
        else:scores=model.predict_proba(np.column_stack((edit_matrix(rows,functions),lg(base),lg(ce))).astype(np.float32))[:,1]
        if not valid_probabilities(scores,len(rows)):raise ValueError('Invalid calibrated scores')
        np.save(parts/f'{part:03d}.npy',scores)
        with db:
            db.executemany('INSERT INTO owners VALUES(?,?)',owners.items())
            db.executemany('INSERT INTO claims VALUES(?,?,?)',[(rows[i][1],rows[i][0],float(scores[i])) for i in np.flatnonzero(scores>=selected['threshold'])])
            db.execute('INSERT INTO partitions VALUES(?,?,?)',(part,recipe,len(owners)))
        progress('cpu_rescoring',completed_partitions=part+1,partitions=128)
    progress('owner_assignment_and_assembly')
    applied=assemble(db,output,selected['threshold'],read(STRUCT/'owner_eval/policy.json')['selected'],load_text(STRUCT/'text_scorer.json'))
    db.close();write(output/'application.json',applied)
    for stage,args in [
        ('stream_validation',['tools/validate_full_submission.py','--test-dir',DATA,'--output',output,'--report',output/'stream_validation.json']),
        ('official_validation',['tools/package_v2_200k_submission.py','official-validate','--validator',DATA.parent.parent/'utils/validate_submission.py',
            '--test-dir',DATA,'--output',output,'--report',output/'official_matching_validation.json'])]:
        progress(stage);command([sys.executable,*args],stage+'.log')
    stream,official=read(output/'stream_validation.json'),read(output/'official_matching_validation.json')
    if not stream['valid'] or stream['references']!=1732544 or not official['valid'] or official['errors'] or official['warnings'] or not official['check_ids']:
        raise ValueError('Peer-based full submission failed validation')
    if not all(r['sha256']['matching_results.tsv']==digest(output/'matching_results.tsv') for r in (stream,official)):
        raise ValueError('Validated matching changed')
    completion=dict(ready_for_submission=True,uploaded_to_portal=False,references=1732544,
        matching=str(output/'matching_results.tsv'),output_sha256=stream['sha256'],development=metrics,
        producer_sha256=digest(identity_path),audit_sha256=digest(audit/'report.json'),parity_sha256=digest(calibration/'parity.json'),
        note='FP16-weight peer scores independently replayed and recalibrated; no French labels/public-score prediction')
    write(output/'completion.json',completion)
    progress('complete',completion=str(output/'completion.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--producer-pid',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--minutes',type=float,default=480)
    p.add_argument('--tag',default='e5_cont1300k');a=p.parse_args()
    if a.minutes<=0:p.error('Finite positive duration required')
    try:run(**vars(a))
    except Exception as error:
        write(a.output/'failure.json',dict(error=str(error),utc=datetime.now(timezone.utc).isoformat(),ready_for_submission=False));raise
