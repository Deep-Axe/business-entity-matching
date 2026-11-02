"""Detached allowlisted v2 experiments with durable exit state and health data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ALLOWED={'tools/v2_sparse_fullgallery.py', 'tools/v2_run_pipeline.py',
         'tools/v2_owner_postprocess.py'}


def safe_output(value):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='artifacts':
        raise ValueError('Output must be an explicit project artifact path')
    return p


def validate_spec(spec):
    argv=spec['argv']
    if not isinstance(argv,list) or len(argv)<3 or argv[0]!='.venv/bin/python' or argv[1] not in ALLOWED:
        raise ValueError('Command is not an allowed v2 experiment')
    if not all(isinstance(v,str) for v in argv):raise ValueError('String arguments required')
    output=safe_output(spec['output'])
    if '--output' not in argv or argv[argv.index('--output')+1]!=str(output):
        raise ValueError('Declared output differs from command')
    return argv,output


def write_json(path,data):
    temporary=path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def run(action,job,spec=None):
    if not job.replace('_','').isalnum():raise ValueError('Unsafe job name')
    root=Path(__file__).resolve().parents[1]
    directory=root/'artifacts/v2_remote_jobs';directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/(job+'.json');exit_path=directory/(job+'.exit.json');log_path=directory/(job+'.log')
    if action=='start':
        if state_path.exists() or log_path.exists():raise ValueError('Existing job must not be duplicated')
        argv,output=validate_spec(json.loads(spec))
        state=dict(job=job,argv=argv,output=str(output),started=time.time())
        write_json(state_path,state)
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with log_path.open('x') as log:
            child=subprocess.Popen(['.venv/bin/python','tools/v2_remote_job.py','execute','--job',job],
                cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['pid']=child.pid
        state['start_ticks']=Path(f'/proc/{child.pid}/stat').read_text().split()[21]
        write_json(state_path,state)
        print(json.dumps(dict(job=job,started=True,pid=child.pid)),flush=True)
        return
    state=json.loads(state_path.read_text())
    if action=='execute':
        argv,_=validate_spec(state)
        result=subprocess.run(argv,cwd=root,check=False)
        write_json(exit_path,dict(returncode=result.returncode,finished=time.time()))
        sys.exit(result.returncode)
    pid=state.get('pid');stat=Path(f'/proc/{pid}/stat')
    fields=stat.read_text().split() if stat.exists() else []
    active=bool(fields and fields[21]==state.get('start_ticks') and fields[2]!='Z')
    if action=='stop' and active:
        if os.getsid(pid)!=pid:raise ValueError('Session ownership changed')
        os.killpg(pid,signal.SIGTERM)
    output=root/safe_output(state['output'])
    def maybe(name):
        p=output/name
        try:return json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:return {'status':'write_in_progress'}
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    terminal=json.loads(exit_path.read_text()) if exit_path.exists() else None
    health=dict(available_memory_mib=memory['MemAvailable']/1024,total_memory_mib=memory['MemTotal']/1024,
        free_disk_gib=shutil.disk_usage(root).free/1024**3,load1=os.getloadavg()[0])
    full_manifest=maybe('manifest.json')
    manifest={k:full_manifest[k] for k in ('complete','full_gallery','rows','gallery_rows','queries',
        'elapsed_seconds','neighbors_sha256','idf_sha256','partition_complete',
        'references','candidate_pairs','partition','partitions') if k in full_manifest}
    report=dict(job=job,active=active,finished=terminal is not None,exit=terminal,
        manifest=manifest,progress=maybe('progress.json'),health=health)
    if terminal and terminal['returncode']!=0:
        report['error_tail']=log_path.read_text(errors='replace')[-2000:]
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['start','status','stop','execute']);p.add_argument('--job',required=True);p.add_argument('--spec')
    run(**vars(p.parse_args()))
