"""Resume read-only EC2 health checks after the user renews AWS CLI login.

Never initiates a login, provisions, extends a timer, or prints credentials.
The absolute previously approved fleet deadline bounds the whole supervisor.
"""
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

from aws_benchmark_ctl import aws
from v3_transfer_data import ROOT, write


def run():
    work = ROOT / 'artifacts/improvement_v2/cloud/health_auth_supervision_20260926'
    receipt = json.loads((ROOT / 'artifacts/improvement_v2/cloud/final_deadline_budget_20260927_v1.json').read_text())
    deadline = datetime.fromisoformat(receipt['deadline_utc'])
    if not receipt['complete'] or not receipt['timers_applied']:
        raise ValueError('Require approved installed deadline')
    while True:
        now = datetime.now(timezone.utc); remaining = (deadline - now).total_seconds()
        if remaining <= 0:
            write(work / 'progress.json', dict(stage='absolute_deadline_reached', utc=now.isoformat()))
            return
        try:
            aws('sts', 'get-caller-identity')
        except Exception as error:
            write(work / 'progress.json', dict(stage='waiting_for_user_aws_login', utc=now.isoformat(),
                api_monitor_healthy=False, error=str(error), deadline_utc=deadline.isoformat(),
                inference_and_remote_timers_unmodified=True))
            time.sleep(min(45, remaining)); continue
        write(work / 'progress.json', dict(stage='read_only_ec2_monitor_running', utc=now.isoformat(),
            deadline_utc=deadline.isoformat()))
        args = [sys.executable, 'tools/v2_ec2_health.py', '--fleets',
                'artifacts/improvement_v2/cloud/fleet_v1', 'artifacts/improvement_v2/cloud/knn_fleet_v1',
                'artifacts/improvement_v2/cloud/submission_fleet_v1', '--output',
                'artifacts/improvement_v2/cloud/submission_health_v1', '--minutes', str(remaining/60), '--interval', '60']
        with (work / 'monitor.log').open('a') as log:
            result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            write(work / 'progress.json', dict(stage='monitor_completed', utc=datetime.now(timezone.utc).isoformat()))
            return
        time.sleep(30)


if __name__ == '__main__':
    run()
