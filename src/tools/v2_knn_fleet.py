"""Finite two-channel kNN orchestration, health checks and small recovery copies.

Only explicitly configured project artifacts/scripts are synchronized. Raw count
blocks remain on temporary workers; reports, global statistics, commit metadata
and final neighbors are copied locally. No credentials are uploaded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from aws_baseline_transfer import ssh
from aws_baseline_jobs import REMOTE
from ber.shards import file_hash
from v2_remote_job import validate_spec,safe_output


def write_json(path,value):
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def specifications():
    specs={}
    for node,channel in enumerate(('word','char')):
        gallery=f'artifacts/improvement_v2/knn/fullgallery_{channel}_v1'
        scored=f'artifacts/improvement_v2/knn/fullgallery_{channel}_scores_v1'
        base=['.venv/bin/python','tools/v2_sparse_fullgallery.py','build','--store','artifacts/train_store_v2',
            '--channel',channel,'--output',gallery,'--workers','2','--block-size','10000','--bins','4194304']
        specs[str(node)]=[
            dict(job=f'knn_{channel}_probe',output=gallery,argv=base+['--stop-after-blocks','5']),
            dict(job=f'knn_{channel}_build',output=gallery,argv=base+['--resume']),
            dict(job=f'knn_{channel}_score',output=scored,argv=['.venv/bin/python','tools/v2_sparse_fullgallery.py',
                'score','--gallery',gallery,'--queries','artifacts/improvement_v2/knn/fullgallery_queries_v1',
                '--output',scored,'--workers','2','--k','50','--query-batch','16'])]
    for items in specs.values():
        for item in items:validate_spec(item)
    return specs


def remote(run,host,action,item):
    argv=['.venv/bin/python','tools/v2_remote_job.py',action,'--job',item['job']]
    if action=='start':argv+=['--spec',json.dumps(item)]
    command='cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)
    result=subprocess.run(ssh(run,host)+[command],capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError(f"node{host['node']} {action} failed: {result.stderr[-1000:]}")
    return json.loads(result.stdout)


def collect(run,host,item):
    safe_output(item['output'])
    dest=run/'collected'/Path(item['output']).name;dest.mkdir(parents=True,exist_ok=True)
    transport=shlex.join(ssh(run,host)[:-1])
    argv=['rsync','-a','--exclude=.*','--include=*/','--include=*.json',
        '--include=neighbors.tsv','--include=idf.npy','--include=df.npy','--exclude=*','-e',transport,
        'ubuntu@'+host['ip']+':'+REMOTE+'/'+item['output']+'/',str(dest)+'/']
    result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(f"node{host['node']} recovery copy failed")
    if (dest/'manifest.json').exists():
        manifest=json.loads((dest/'manifest.json').read_text())
        for filename,key in (('neighbors.tsv','neighbors_sha256'),('idf.npy','idf_sha256'),('df.npy','df_sha256')):
            if key in manifest and file_hash(dest/filename)!=manifest[key]:raise ValueError('Collected artifact hash mismatch')
    return str(dest)


def sync(run,state):
    files=['tools/v2_remote_job.py','tools/v2_sparse_fullgallery.py',
           'artifacts/improvement_v2/knn/fullgallery_queries_v1']
    def one(host):
        result=subprocess.run(['rsync','-aR','--exclude=__pycache__','-e',shlex.join(ssh(run,host)[:-1]),
            *files,'ubuntu@'+host['ip']+':'+REMOTE+'/'],capture_output=True,text=True,timeout=180)
        if result.returncode:raise RuntimeError('Allowlisted synchronization failed')
        return dict(node=host['node'],synchronized=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one,state['hosts']):print(json.dumps(result),flush=True)
    write_json(run/'knn_specs.json',specifications())
    write_json(run/'knn_deployment.json',{name:file_hash(name) for name in files[:2]})


def monitor(run,state,minutes,interval,backup_interval):
    specs=json.loads((run/'knn_specs.json').read_text())
    checkpoint=run/'knn_orchestration.json'
    orchestration=json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        'phase':{str(h['node']):0 for h in state['hosts']},'done':{},'started_jobs':[]}
    started=last_backup=time.monotonic();failures=0
    while time.monotonic()-started<minutes*60:
        reports=[]
        for host in state['hosts']:
            node=str(host['node']);phase=orchestration['phase'][node];item=specs[node][phase]
            if orchestration['done'].get(node):continue
            try:
                if item['job'] not in orchestration['started_jobs']:
                    # Recover a lost launch response without duplicating an existing job.
                    try:remote(run,host,'status',item)
                    except RuntimeError:remote(run,host,'start',item)
                    orchestration['started_jobs'].append(item['job']);write_json(checkpoint,orchestration)
                status=remote(run,host,'status',item)
                reports.append(dict(node=host['node'],phase=phase,**status))
                if not status['active'] and not status['finished']:
                    raise RuntimeError('Experiment exited without durable exit state; inspect before restart')
                if status['health']['free_disk_gib']<5 or status['health']['available_memory_mib']<512:
                    remote(run,host,'stop',item)
                    raise RuntimeError('Worker resource safety threshold crossed; job stopped')
                if status['finished']:
                    if status['exit']['returncode']!=0:raise RuntimeError('Experiment failed: '+status.get('error_tail',''))
                    collect(run,host,item)
                    if phase==0:
                        p=status['progress']
                        if p.get('committed_blocks',0)<5:raise ValueError('Timing probe did not commit five blocks')
                    elif phase==1:
                        if not status['manifest'].get('full_gallery') or status['manifest'].get('rows')!=10320219:
                            raise ValueError('Incomplete full-gallery build')
                    else:
                        m=status['manifest']
                        if not m.get('full_gallery') or m.get('gallery_rows')!=10320219 or m.get('queries')!=800:
                            raise ValueError('Incomplete full-gallery score')
                    if phase<2:orchestration['phase'][node]=phase+1
                    else:orchestration['done'][node]=True
                    write_json(checkpoint,orchestration)
            except (RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
                reports.append(dict(node=host['node'],phase=phase,error=str(error)))
        summary=dict(utc=datetime.now(timezone.utc).isoformat(),nodes=reports,orchestration=orchestration)
        write_json(run/'knn_progress.json',summary)
        with (run/'knn_progress.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        print(json.dumps(summary),flush=True)
        if any('error' in r for r in reports):
            failures+=1
            if failures==2:
                # Safe helper changes only this run's recorded single-client /32.
                from refresh_baseline_ssh_source import run as refresh
                refresh(run)
            if failures>=3:raise RuntimeError('Repeated experiment failures; inspect progress, no blind restart')
        else:failures=0
        if time.monotonic()-last_backup>=backup_interval:
            backups=[]
            for host in state['hosts']:
                node=str(host['node']);item=specs[node][orchestration['phase'][node]]
                if item['job'] in orchestration['started_jobs']:
                    try:backups.append(dict(node=host['node'],success=True,path=collect(run,host,item)))
                    except Exception as error:backups.append(dict(node=host['node'],success=False,error=str(error)))
            write_json(run/'knn_backup_status.json',dict(utc=datetime.now(timezone.utc).isoformat(),nodes=backups))
            last_backup=time.monotonic()
        if all(orchestration['done'].get(str(h['node'])) for h in state['hosts']):return
        time.sleep(interval)
    print('Finite monitor deadline reached; instance shutdown protection remains active',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['sync','monitor']);p.add_argument('--run',required=True)
    p.add_argument('--minutes',type=float,default=240);p.add_argument('--interval',type=float,default=60)
    p.add_argument('--backup-interval',type=float,default=300)
    a=p.parse_args();r=Path(a.run).resolve();s=json.loads((r/'state.json').read_text())
    if not s.get('hosts_pinned') or not s.get('nodes_configured'):raise ValueError('Fleet not configured/pinned')
    if a.action=='sync':sync(r,s)
    else:monitor(r,s,a.minutes,a.interval,a.backup_interval)
