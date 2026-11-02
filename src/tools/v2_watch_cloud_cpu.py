"""Finite status/backup monitor for the already launched, isolated 200k fit.

No provisioning, timer changes, termination, release markers or inference jobs.
Only the explicit small retained outputs are collected; no large matrix copies.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time

from aws_baseline_transfer import ssh

REMOTE = '/home/ubuntu/amazon-ml-baseline'
CPU = 'artifacts/improvement_v2/cpu'
JOB = 'v2_cloud_cpu_200k_v1'


def probe(run, host):
    code = """import json,pathlib,shutil
root=pathlib.Path('/home/ubuntu/amazon-ml-baseline')
state=json.loads((root/'artifacts/remote_jobs/v2_cloud_cpu_200k_v1.json').read_text())
p=pathlib.Path('/proc')/str(state['pid'])/'stat';active=False
if p.exists():
 fields=p.read_text().split();active=fields[21]==state['start_ticks'] and fields[2]!='Z'
r=root/'artifacts/improvement_v2/cpu/cloud200k_receipts_v1'
stage='starting'
for name in ('prepare','quantize','fit'):
 if (r/(name+'_start.json')).exists():stage=name
 if (r/(name+'_complete.json')).exists():stage=name+'_complete'
out=dict(active=active,stage=stage,complete=(r/'complete.json').exists(),failure=(r/'failure.json').exists())
progress=root/'artifacts/improvement_v2/cpu/prepared_gate0_train_200k_v1/progress.json'
if progress.exists():out['preparation']=json.loads(progress.read_text())
if out['failure']:out['failure_detail']=json.loads((r/'failure.json').read_text())
fitlog=r/'fit.log'
if fitlog.exists():out['fit_log_tail']=fitlog.read_text()[-1200:]
memory=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
out['available_gib']=int(memory['MemAvailable'].split()[0])/1024**2
out['disk_free_gib']=shutil.disk_usage(root).free/2**30
print(json.dumps(out))"""
    result = subprocess.run(ssh(run, host) + [shlex.join(['python3', '-c', code])],
        capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError('Scoped cloud CPU status check failed')
    return json.loads(result.stdout)


def collect(run, host, status):
    transport = shlex.join(ssh(run, host)[:-1])
    routes = [(CPU + '/cloud200k_receipts_v1/', Path(CPU) / 'cloud200k_receipts_v1')]
    if status['stage'].startswith('fit') or status['complete']:
        routes.append((CPU + '/fit_gate0_200k_v1/', Path(CPU) / 'fit_gate0_200k_v1'))
    provenance = Path(CPU) / 'cloud200k_provenance_v1'
    provenance.mkdir(parents=True, exist_ok=True)
    if status['stage'] != 'starting' and status['stage'] != 'prepare':
        routes.append((CPU + '/prepared_gate0_train_200k_v1/manifest.json', provenance / 'prepared_manifest.json'))
    if status['stage'].startswith('fit') or status['complete']:
        routes.append((CPU + '/quantized_gate0_200k_v1/enhanced/manifest.json', provenance / 'quantized_manifest.json'))
    for remote, local in routes:
        if remote.endswith('/'):
            local.mkdir(parents=True, exist_ok=True)
            destination = str(local) + '/'
        else:
            destination = str(local)
        result = subprocess.run(['rsync', '-a', '--exclude=.*', '-e', transport,
            'ubuntu@' + host['ip'] + ':' + REMOTE + '/' + remote, destination],
            capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Scoped cloud CPU artifact collection failed')


def run(run, output, minutes=180, interval=60, backup_interval=300):
    if not 1 <= minutes <= 240 or not 15 <= interval <= 60 or backup_interval < interval:
        raise ValueError('Finite monitoring limits exceeded')
    run, output = Path(run).resolve(), Path(output)
    state = json.loads((run / 'state.json').read_text())
    if not state.get('hosts_pinned'):
        raise ValueError('Authenticated host keys required')
    host = next(h for h in state['hosts'] if h['node'] == 0)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    copied = started - backup_interval
    failures = 0
    while time.monotonic() - started < minutes * 60:
        try:
            status = probe(run, host)
            status['utc'] = datetime.now(timezone.utc).isoformat()
            (output / 'status.json').write_text(json.dumps(status, indent=2))
            with (output / 'status.jsonl').open('a') as stream:
                stream.write(json.dumps(status) + '\n')
            print(json.dumps(status), flush=True)
            if time.monotonic() - copied >= backup_interval or status['complete'] or status['failure']:
                collect(run, host, status)
                copied = time.monotonic()
                (output / 'backup.json').write_text(json.dumps(dict(success=True, utc=datetime.now(timezone.utc).isoformat())))
            if status['failure'] or not status['active'] and not status['complete']:
                raise RuntimeError('Cloud training exited without verified completion')
            if status['complete'] and not status['active']:
                print('Cloud training exited successfully and retained outputs collected', flush=True)
                return
            failures = 0
        except Exception as error:
            failures += 1
            print(json.dumps(dict(monitor_error=str(error), consecutive_failures=failures)), flush=True)
            if failures >= 3:
                raise
        time.sleep(interval)
    raise TimeoutError('Finite cloud training monitor deadline reached; inspect job without extending timers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--minutes', type=int, default=180)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--backup-interval', type=int, default=300)
    run(**vars(parser.parse_args()))
