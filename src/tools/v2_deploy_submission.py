"""Deploy one reviewed test payload to explicitly tagged, host-pinned workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
from aws_baseline_transfer import ssh
from aws_baseline_fleet import TAG
from aws_benchmark_ctl import aws,PROFILE

def deploy(run,source_run):
    run,source_run=map(Path,(run,source_run))
    state=json.loads((run/'state.json').read_text())
    source=json.loads((source_run/'state.json').read_text())
    if not state.get('hosts_pinned') or not source.get('payload_uploaded'):
        raise ValueError('Pinned hosts and uploaded reviewed payload required')
    tags=aws('s3api','get-bucket-tagging','--bucket',source['bucket'],region=source['region'])
    if {t['Key']:t['Value'] for t in tags['TagSet']}.get(TAG)!=source['run_id']:
        raise ValueError('Payload ownership mismatch')
    signed=subprocess.run(['aws','s3','presign',f"s3://{source['bucket']}/payload-v1.tar.gz",
        '--expires-in','7200','--profile',PROFILE,'--region',source['region']],capture_output=True,text=True)
    if signed.returncode:raise RuntimeError('Scoped signing failed')
    script='\n'.join(['set -euo pipefail','cd /home/ubuntu/amazon-ml-baseline',
        'test -f bootstrap.ready','test ! -f submission_v2_payload.ready',
        'curl --fail --silent --show-error --retry 3 '+shlex.quote(signed.stdout.strip())+' -o submission_payload_v1.tar.gz',
        "printf '%s  submission_payload_v1.tar.gz\\n' "+shlex.quote(source['payload_sha256'])+' | sha256sum --check --status',
        'tar -xzf submission_payload_v1.tar.gz',
        'if ! test -x .venv/bin/python; then python3 -m venv .venv; .venv/bin/pip install --disable-pip-version-check -q -e code/business_entity_resolution; fi',
        "printf '%s\\n' "+shlex.quote(source['payload_sha256'])+' > submission_v2_payload.ready',''])
    def one(host):
        result=subprocess.run(ssh(run,host)+['bash','-s'],input=script,capture_output=True,text=True,timeout=1200)
        report=dict(node=host['node'],complete=result.returncode==0,payload_sha256=source['payload_sha256'])
        (run/f"submission_setup_node_{host['node']}.json").write_text(json.dumps(report)+'\n')
        if result.returncode:
            private=run/f"submission_setup_node_{host['node']}.private.log"
            private.write_text(result.stderr);os.chmod(private,0o600)
        print(json.dumps(report),flush=True)
        return report
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,state['hosts']))
    if not all(v['complete'] for v in results):raise RuntimeError('Deployment incomplete; inspect private logs')
    # Never copy the source bucket into another fleet's lifecycle state.
    (run/'submission_deployment.json').write_text(json.dumps(dict(complete=True,
        payload_sha256=source['payload_sha256'],source_run=source['run_id'],hosts=len(results)),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--source-run',required=True)
    deploy(**vars(p.parse_args()))
