"""Finite read-only EC2 system/instance status monitor for explicitly owned fleets."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from aws_benchmark_ctl import aws

def run(fleets,output,minutes=690,interval=60):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    labels={}
    for directory in fleets:
        state=json.loads((Path(directory)/'state.json').read_text())
        if state['region']!='ap-southeast-2':raise ValueError('Unexpected region')
        for h in state['hosts']:labels[h['instance_id']]=f'{Path(directory).name}_node{h["node"]}'
    start=time.monotonic();failures=0
    while time.monotonic()-start<minutes*60:
        try:
            result=aws('ec2','describe-instance-status','--include-all-instances','--instance-ids',*labels)
            nodes=[dict(node=labels[v['InstanceId']],state=v['InstanceState']['Name'],
                system=v.get('SystemStatus',{}).get('Status'),instance=v.get('InstanceStatus',{}).get('Status'))
                for v in result['InstanceStatuses']]
            report=dict(utc=datetime.now(timezone.utc).isoformat(),nodes=nodes)
            if len(nodes)!=len(labels):raise ValueError('Missing EC2 health response')
            failures=0
        except Exception as error:
            failures+=1;report=dict(utc=datetime.now(timezone.utc).isoformat(),error=str(error),failures=failures)
        temp=output/'health.pending.json';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output/'health.json')
        with (output/'health.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)
        if report.get('nodes') and all(n['state']=='terminated' for n in report['nodes']):return
        if failures>=3:raise RuntimeError('Repeated EC2 health access failure; monitoring not silently healthy')
        time.sleep(interval)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fleets',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--minutes',type=float,default=690);p.add_argument('--interval',type=float,default=60)
    run(**vars(p.parse_args()))
