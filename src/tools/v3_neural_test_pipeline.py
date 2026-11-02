"""Bounded, resumable deployment of an independently audited CE/edit stacker.

Waits for an idle GPU; never terminates another job. Validates the checkpoint
against saved tuning probabilities, checks every routed test pair against the
committed 100k candidates, and stores claims on disk. No leaderboard uploads.
Trusted local pickle inputs only. Run under the shared CPU/memory cgroup.
"""
import argparse
import ast
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import subprocess
import sys
import time

import joblib
import numpy as np

from ber.data import read_tsv,write_tsv
from ber.store import connect,lookup
from v3_owner_assignment import load_text,text_logit,winner,open_claims
from v3_transfer_data import ROOT,digest,write

SCRATCH=Path('work/ce')
CONTROLLER=ROOT/'artifacts/improvement_v2/inference/submission_controller_v1'
BASELINE=ROOT/'artifacts/duf-fel_full_output_v2_100k'
DATA=ROOT/'6ab10eb3b23ba_student_resource/student_resource/dataset/test'
STRUCT=ROOT/'artifacts/improvement_v3/structural_v1'


def read(path):return json.loads(Path(path).read_text())


def lg(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6);return np.log(p/(1-p))


def feature_functions():
    """Load only the locally authored feature definitions, not analysis scripts."""
    namespace={};hashes={}
    for name in ('editops.py','editops2.py'):
        path=SCRATCH.parent/name;hashes[str(path)]=digest(path)
        tree=ast.parse(path.read_text());nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='con' for t in node.targets):break
            if isinstance(node,ast.ImportFrom) and node.module=='editops':continue
            if isinstance(node,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef)):nodes.append(node)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    # editops2 deliberately supplies the final nums() helper, just like the
    # original feature-generation script. Exact tuning-array parity is required.
    return namespace['feats'],namespace['offset_feats'],hashes


def edit_matrix(rows,functions):
    feats,offset,_=functions
    return np.asarray([feats(r[5],r[6],r[7],r[8])+offset(r[6],r[8]) for r in rows],dtype=np.float32).reshape(-1,28)


def gpu_processes():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,check=True,timeout=20)
    return {int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()}


def resources(min_ram_gib=3,min_disk_gib=6):
    values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    if values['MemAvailable']<min_ram_gib*1024**2:raise RuntimeError('Low available RAM; preserving checkpointed work')
    if shutil.disk_usage(ROOT).free<min_disk_gib*1024**3:raise RuntimeError('Low disk; preserving baseline and completed parts')


def verify_route(rows,expected,owners):
    seen=set()
    for row in rows:
        key=(row[0],row[1])
        if key in seen or key not in expected:raise ValueError('Duplicate or non-candidate routed pair')
        seen.add(key)
        if abs(float(row[4])-expected[key])>1e-8:raise ValueError('Routed base probability changed')
        if row[3]!=owners[row[0]]:raise ValueError('Country mismatch')
    if seen!=set(expected):raise ValueError('Routed candidate coverage incomplete')


def partition_inputs(part,entry):
    directory=Path(entry['directory']);manifest=read(directory/'manifest.json');cfg=read(directory/'config.json')
    if not manifest['complete'] or not manifest['partition_complete'] or cfg['partition']!=part:
        raise ValueError('Source partition incomplete/wrong')
    if cfg['model_sha256']!=digest(ROOT/'artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced.cbm'):
        raise ValueError('Wrong source model')
    expected={};owners={}
    for shard in sorted((directory/'shards').glob('[0-9]*')):
        done=read(shard/'done.json')
        if done['config_sha256']!=digest(directory/'config.json'):raise ValueError('Source shard identity changed')
        for name in ('pair_ids.npz','probabilities.npy'):
            if digest(shard/name)!=done['output_sha256'][name]:raise ValueError('Source shard checksum failed')
        with np.load(shard/'pair_ids.npz',allow_pickle=False) as z:
            refs=z['reference_ids'];countries=z['countries'];q=z['query_ids'];targets=z['target_ids']
        p=np.load(shard/'probabilities.npy',allow_pickle=False)
        if len(p)!=len(q) or len(q)!=len(targets):raise ValueError('Source pair alignment changed')
        for owner,country in zip(refs,countries):
            if str(owner) in owners:raise ValueError('Duplicate partition owner')
            owners[str(owner)]=str(country)
        for i in np.flatnonzero(p>=.01):
            key=(str(refs[q[i]]),str(targets[i]))
            if key in expected:raise ValueError('Duplicate candidate')
            expected[key]=float(p[i])
    if len(owners)!=entry['references']:raise ValueError('Partition owner coverage incomplete')
    return expected,owners


def assemble(db,output,threshold,rule,text_model):
    if db.execute('SELECT count(*) FROM partitions').fetchone()[0]!=128:
        raise ValueError('Refusing partial test output')
    if db.execute('SELECT count(*) FROM owners').fetchone()[0]!=1732544:
        raise ValueError('Refusing incomplete S1 coverage')
    db.execute('CREATE TABLE IF NOT EXISTS keep(target TEXT PRIMARY KEY,owner TEXT)')
    source=connect(ROOT/'artifacts/test_store_v2');source.execute('PRAGMA cache_size=-8192')
    changed=abstained=conflicts=0
    for (target,) in db.execute('SELECT target FROM claims GROUP BY target HAVING count(*)>1'):
        edges=list(db.execute('SELECT owner,p FROM claims WHERE target=? ORDER BY owner',(target,)))
        owners=lookup(source,'anchors','entity_id',[r for r,_ in edges]);row=lookup(source,'records','entity_id',[target])[target]
        text=[text_logit(owners[r],row,text_model) for r,_ in edges]
        choice=winner([p for _,p in edges],text,**rule)
        db.execute('INSERT OR REPLACE INTO keep VALUES(?,?)',(target,None if choice is None else edges[choice][0]))
        changed+=len(edges)-(choice is not None);abstained+=choice is None;conflicts+=1
        if conflicts%1000==0:db.commit()
    db.commit();source.close()
    counts=dict(references=0,predicted_pairs=0,empty_owners=0)
    def rows():
        for ref in read_tsv(DATA/'test_source1.tsv'):
            owner=ref['entity_id'];known=db.execute('SELECT country FROM owners WHERE owner=?',(owner,)).fetchone()
            if known is None or known[0]!=ref['country']:raise ValueError('S1 coverage/country changed')
            kept=[r[0] for r in db.execute('SELECT c.target FROM claims c LEFT JOIN keep k ON c.target=k.target '
                'WHERE c.owner=? AND (k.target IS NULL OR k.owner=c.owner) ORDER BY c.target',(owner,))]
            counts['references']+=1;counts['predicted_pairs']+=len(kept);counts['empty_owners']+=not kept
            yield dict(source1_entity_id=owner,matched_entity_ids=','.join(kept))
    tmp=output/'matching_results.tsv.tmp';write_tsv(tmp,rows(),['source1_entity_id','matched_entity_ids'])
    if counts['references']!=1732544:raise ValueError('Incomplete final output')
    tmp.replace(output/'matching_results.tsv')
    candidate=BASELINE/'candidate_pairs.tsv';assembly=read(BASELINE/'assembly.json')
    if not assembly['complete_submission'] or digest(candidate)!=assembly['output_sha256']['candidate_pairs.tsv']:
        raise ValueError('Full baseline candidates changed')
    if not (output/'candidate_pairs.tsv').exists():os.link(candidate,output/'candidate_pairs.tsv')
    return dict(**counts,removed_edges=changed,abstained_targets=abstained,conflicting_targets_before=conflicts,
                conflicting_targets_after=0,threshold=threshold,matching_sha256=digest(output/'matching_results.tsv'))


def run(audit,output,minutes=390,idle_seconds=60,check_only=False):
    audit,output=Path(audit),Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    end=time.monotonic()+minutes*60;report=read(audit/'report.json');selected=report['selected']
    if not report['complete'] or not selected['name'].startswith('ops_ce_'):raise ValueError('Audited stacker required')
    if report['models'][selected['name']]['development']['macro_f05']<=report['baseline_replay']['macro_f05']:
        raise ValueError('Audited variant is weaker than baseline')
    for path,expected in report['inputs_sha256'].items():
        if digest(path)!=expected:raise ValueError('An audited input changed: '+path)
    tag=report['tag'];weights=SCRATCH/tag/'model.pt';stack_path=audit/(selected['name']+'.joblib')
    functions=feature_functions();stack=joblib.load(stack_path)
    recipe=dict(audit_sha256=digest(audit/'report.json'),checkpoint_sha256=digest(weights),
        stacker_sha256=digest(stack_path),selected=selected,feature_source_sha256=functions[2],
        script_sha256=digest(Path(__file__)),tag=tag,route_minimum=.01,
        gpu_worker_sha256=digest(ROOT/'tools/v3_ce_score_partition.py'),
        policy_sha256=digest(STRUCT/'owner_eval/policy.json'),text_sha256=digest(STRUCT/'text_scorer.json'))
    if (output/'identity.json').exists():
        if read(output/'identity.json')!=recipe:raise ValueError('Pipeline identity changed')
    else:write(output/'identity.json',recipe)
    retained=output/'reproduction';retained.mkdir(exist_ok=True)
    for source,name in ((weights,'model.pt'),(stack_path,'stacker.joblib')):
        if not (retained/name).exists():os.link(source,retained/name)
        if digest(retained/name)!=digest(source):raise ValueError('Retained model differs')
    weights=retained/'model.pt'
    for source in [Path(__file__),ROOT/'tools/v3_ce_score_partition.py',*(Path(p) for p in functions[2])]:
        if not (retained/source.name).exists():shutil.copyfile(source,retained/source.name)
        if digest(retained/source.name)!=digest(source):raise ValueError('Retained source changed')
    def progress(stage,**extra):
        r=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),ready_for_submission=False,uploaded_to_portal=False)
        r.update(extra);write(output/'progress.json',r);print(json.dumps(r),flush=True)
    def alive():
        if time.monotonic()>=end:raise TimeoutError('Finite neural deployment window expired')
        resources()
    # Ensure locally defined features exactly match those fitted in the audit.
    for split in ('tune','dev'):
        rows=pickle.load((SCRATCH/f'{split}.pkl').open('rb'))
        if not np.array_equal(edit_matrix(rows,functions),np.load(SCRATCH/f'{split}.pkl.ops.npy',allow_pickle=False)):
            raise ValueError('Edit-feature replay differs from audited features')
    if check_only:
        progress('cpu_preflight_passed',gpu_started=False);return
    quiet=None
    while True:
        alive();busy=gpu_processes()
        if busy:quiet=None
        elif quiet is None:quiet=time.monotonic()
        if quiet is not None and time.monotonic()-quiet>=idle_seconds:break
        progress('waiting_for_idle_gpu',compute_processes=len(busy));time.sleep(15)
    # A cooperating future worker can share this lock. Noncooperating processes
    # are checked before every batch; this job yields by exiting, never kills them.
    gpu_lock=(ROOT/'artifacts/improvement_v3/local_gpu.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model_dir=ROOT/'artifacts/improvement_v3/transfer_pilot_v1/models/e5'
    def predict(path):
        alive()
        env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'.venv-cascade-v2/lib/python3.10/site-packages/nvidia/nvjitlink/lib'),
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        # Keep sklearn/numpy CPU dependencies separate from the running trainer's
        # existing torch venv. No environment is changed or packages installed.
        destination=output/'last_ce_scores.npy'
        with (output/'gpu_scoring.log').open('a') as log:
            subprocess.run([str(ROOT/'.venv-cascade-v2/bin/python'),str(ROOT/'tools/v3_ce_score_partition.py'),
                '--model-dir',str(model_dir),'--weights',str(weights),'--input',str(path),'--output',str(destination)],
                env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
        return np.load(destination,allow_pickle=False)
    # Binding a checkpoint by filename is insufficient; replay actual predictions.
    progress('checkpoint_preflight')
    p=predict(SCRATCH/'tune.pkl')
    saved=np.load(SCRATCH/tag/'tune_pce.npy',allow_pickle=False)
    delta=float(np.max(np.abs(p-saved)))
    if delta>2e-4:raise ValueError(f'Checkpoint does not reproduce audited CE probabilities: {delta}')
    write(output/'checkpoint_preflight.json',dict(max_absolute_delta=delta,pairs=len(p),checkpoint_sha256=digest(weights)))
    queue=read(CONTROLLER/'queue.json')
    if set(queue['complete'])!=set(map(str,range(128))) or queue.get('assigned'):
        raise ValueError('Complete baseline controller required')
    db=open_claims(output/'cache');threshold=selected['threshold'];parts=output/'scores';parts.mkdir(exist_ok=True)
    for part in range(128):
        alive();path=SCRATCH/'test'/f'part_{part:03d}.pkl'
        part_identity=dict(input_sha256=digest(path),recipe_sha256=digest(output/'identity.json'))
        identity=json.dumps(part_identity,sort_keys=True)
        old=db.execute('SELECT identity FROM partitions WHERE partition=?',(part,)).fetchone()
        if old:
            if old[0]!=identity:raise ValueError('Completed input changed')
            continue
        expected,owners=partition_inputs(part,queue['complete'][str(part)])
        rows=pickle.load(path.open('rb'));verify_route(rows,expected,owners);del expected
        array=parts/f'{part:03d}.npy';ce_array=parts/f'{part:03d}.ce.npy';receipt=parts/f'{part:03d}.json'
        if receipt.exists():
            done=read(receipt)
            if (done['identity']!=part_identity or digest(array)!=done['score_sha256']
                    or digest(ce_array)!=done['ce_sha256']):raise ValueError('Saved scores changed')
            scores=np.load(array,allow_pickle=False)
        else:
            pce=predict(path);ops=edit_matrix(rows,functions)
            X=np.column_stack((ops,lg([r[4] for r in rows]),lg(pce))).astype(np.float32)
            scores=stack.predict_proba(X)[:,1]
            if digest(path)!=part_identity['input_sha256']:raise ValueError('Input changed during scoring')
            np.save(parts/f'{part:03d}.tmp.npy',scores);(parts/f'{part:03d}.tmp.npy').replace(array)
            np.save(parts/f'{part:03d}.ce.tmp.npy',pce);(parts/f'{part:03d}.ce.tmp.npy').replace(ce_array)
            write(receipt,dict(identity=part_identity,pairs=len(rows),score_sha256=digest(array),ce_sha256=digest(ce_array)))
        if scores.shape!=(len(rows),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
            raise ValueError('Invalid model scores')
        with db:
            db.executemany('INSERT INTO owners VALUES(?,?)',owners.items())
            db.executemany('INSERT INTO claims VALUES(?,?,?)',[(rows[i][1],rows[i][0],float(scores[i])) for i in np.flatnonzero(scores>=threshold)])
            db.execute('INSERT INTO partitions VALUES(?,?,?)',(part,identity,len(owners)))
        progress('test_scoring',completed_partitions=part+1,partitions=128,pairs_this_partition=len(rows))
    # Release VRAM before CPU assembly/validation and let another GPU job proceed.
    fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()
    progress('owner_assignment_and_assembly')
    applied=assemble(db,output,threshold,read(STRUCT/'owner_eval/policy.json')['selected'],load_text(STRUCT/'text_scorer.json'))
    db.close();write(output/'application.json',applied)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    for stage,args in [
        ('stream_validation',['tools/validate_full_submission.py','--test-dir',DATA,'--output',output,'--report',output/'stream_validation.json']),
        ('official_validation',['tools/package_v2_200k_submission.py','official-validate','--validator',DATA.parent.parent/'utils/validate_submission.py',
          '--test-dir',DATA,'--output',output,'--report',output/'official_matching_validation.json'])]:
        alive();progress(stage)
        with (output/(stage+'.log')).open('a') as log:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.monotonic())
    stream=read(output/'stream_validation.json');official=read(output/'official_matching_validation.json')
    if not stream['valid'] or stream['references']!=1732544 or not official['valid'] or official['errors'] or official['warnings'] or not official['check_ids']:
        raise ValueError('Submission validation failed')
    if stream['sha256']['matching_results.tsv']!=digest(output/'matching_results.tsv') or official['sha256']['matching_results.tsv']!=digest(output/'matching_results.tsv'):
        raise ValueError('Validated matching changed')
    completion=dict(ready_for_submission=True,uploaded_to_portal=False,matching=str(output/'matching_results.tsv'),
        identity_sha256=digest(output/'identity.json'),output_sha256=stream['sha256'],references=1732544,
        development=report['models'][selected['name']]['development'],
        note='No French labels; no measured leaderboard score; baseline outputs unchanged')
    write(output/'completion.json',completion);progress('complete',**completion)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=390);p.add_argument('--idle-seconds',type=int,default=60)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.minutes<=0 or args.idle_seconds<30:p.error('Positive finite runtime and at least 30 seconds GPU-idle guard required')
    try:run(**vars(args))
    except Exception as error:
        write(args.output/'failure.json',dict(error=str(error),utc=datetime.now(timezone.utc).isoformat(),ready_for_submission=False))
        raise
