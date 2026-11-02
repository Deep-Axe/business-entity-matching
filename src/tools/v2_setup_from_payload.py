"""Configure a new tagged fleet from an existing reviewed private payload.

Borrowed bucket ownership is never copied into the destination's lifecycle state.
Only a scoped signed GET reaches workers; account credentials remain local.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor

from aws_benchmark_ctl import aws,PROFILE,REGION
from aws_baseline_fleet import TAG,save
from aws_baseline_transfer import ssh


def configure(run,source_run,payload_sha256):
    run,source_run=Path(run).resolve(),Path(source_run).resolve()
    state=json.loads((run/'state.json').read_text())
    source=json.loads((source_run/'state.json').read_text())
    if not state.get('hosts_pinned') or not source.get('payload_uploaded'):
        raise ValueError('Pinned destination and completed source upload required')
    if source['payload_sha256']!=payload_sha256 or state['region']!=source['region']:
        raise ValueError('Payload hash/region mismatch')
    tags=aws('s3api','get-bucket-tagging','--bucket',source['bucket'])
    if {t['Key']:t['Value'] for t in tags['TagSet']}.get(TAG)!=source['run_id']:
        raise ValueError('Source bucket ownership mismatch')
    signed=subprocess.run(['aws','s3','presign',f"s3://{source['bucket']}/payload-v1.tar.gz",
        '--expires-in','7200','--profile',PROFILE,'--region',REGION],capture_output=True,text=True)
    if signed.returncode:raise RuntimeError('Scoped payload signing failed')
    script='\n'.join(['set -euo pipefail','cd /home/ubuntu/amazon-ml-baseline',
        'test -f bootstrap.ready','test ! -f setup.complete',
        'curl --fail --silent --show-error --retry 3 '+shlex.quote(signed.stdout.strip())+' -o payload.tar.gz',
        "printf '%s  payload.tar.gz\\n' "+shlex.quote(payload_sha256)+' | sha256sum --check --status',
        'tar -xzf payload.tar.gz','python3 -m venv .venv',
        '.venv/bin/pip install --disable-pip-version-check -q -e code/business_entity_resolution',
        'touch setup.complete',''])
    def node(host):
        result=subprocess.run(ssh(run,host)+['bash','-s'],input=script,capture_output=True,text=True,timeout=900)
        report=dict(node=host['node'],setup_complete=result.returncode==0)
        (run/f"setup_node_{host['node']}.json").write_text(json.dumps(report)+'\n')
        if result.returncode:
            private=run/f"setup_node_{host['node']}.private.log"
            private.write_text(result.stderr);os.chmod(private,0o600)
        print(json.dumps(report),flush=True)
        return report
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(node,state['hosts']))
    if not all(r['setup_complete'] for r in results):raise RuntimeError('Setup incomplete; private logs retained')
    state.update(nodes_configured=True,payload_origin_run=source['run_id'],
        payload_sha256=payload_sha256,payload_bytes=source['payload_bytes'],
        borrowed_payload_bucket_not_owned=True)
    save(run,state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','source-run','payload-sha256'):p.add_argument('--'+k,required=True)
    configure(**vars(p.parse_args()))
