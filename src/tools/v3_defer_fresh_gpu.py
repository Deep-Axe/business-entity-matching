"""Resume our cached feature job after the identified peer overnight queue exits.

Never signal or modify the peer process. PID start ticks prevent PID reuse from
extending the wait. The feature job independently checks CUDA activity and its
exclusive advisory lease when resumed. This supervisor has a finite lifetime.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('.').resolve()
WORK = ROOT / 'artifacts/improvement_v3/fresh_gpu_deferred_20260926'
SERVICE = 'amazon-ml-fresh-scale-features-20260926t1340.service'
FIT_SERVICE = 'amazon-ml-fresh-scale-fit-20260926t1401.service'
FEATURE_COMPLETION = ROOT / 'artifacts/improvement_v3/fresh_scale_features_v1/completion.json'


def start_ticks(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(')', 1)[1].split()[19]


def write_progress(value):
    tmp = WORK / 'progress.tmp.json'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(WORK / 'progress.json')


def ensure_service(service, script, high, maximum, runtime):
    """A stopped transient systemd unit can be garbage-collected; recreate it."""
    state = subprocess.run(['systemctl', '--user', 'show', service, '-p', 'LoadState', '--value'],
                           capture_output=True, text=True, timeout=15)
    if state.returncode == 0 and state.stdout.strip() == 'loaded':
        subprocess.run(['systemctl', '--user', 'start', service], check=True, timeout=30)
        return
    subprocess.run([
        'systemd-run', '--user', '--unit=' + service, '--slice=amazon-ml.slice',
        '--property=WorkingDirectory=' + str(ROOT), '--property=MemoryHigh=' + high,
        '--property=MemoryMax=' + maximum, '--property=MemorySwapMax=128M',
        '--property=Nice=19', '--property=RuntimeMaxSec=' + str(runtime),
        '/usr/bin/env', '-u', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE=1',
        'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        str(ROOT / '.venv/bin/python'), str(ROOT / 'tools' / script)
    ], check=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--start-ticks', required=True)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 12 * 3600
    while start_ticks(args.pid) == args.start_ticks:
        if time.monotonic() >= deadline:
            write_progress(dict(stage='wait_expired_no_restart', peer_pid=args.pid))
            return
        write_progress(dict(stage='waiting_for_peer_queue_exit', peer_pid=args.pid,
                            peer_start_ticks=args.start_ticks, peer_modified=False))
        time.sleep(30)
    ensure_service(SERVICE, 'v3_fresh_cloud_features.py', '3G', '4G', 22000)
    write_progress(dict(stage='cached_feature_service_restarted', service=SERVICE,
                        peer_modified=False, gpu_exclusivity_checked_by_feature_service=True))
    deadline = time.monotonic() + 6 * 3600 + 120
    while not FEATURE_COMPLETION.exists():
        if time.monotonic() >= deadline:
            write_progress(dict(stage='feature_completion_wait_expired', service=SERVICE,
                                fit_started=False, peer_modified=False))
            return
        state = subprocess.run(['systemctl', '--user', 'is-active', SERVICE],
                               capture_output=True, text=True, timeout=15)
        if state.stdout.strip() not in ('active', 'activating'):
            write_progress(dict(stage='feature_job_stopped_before_completion',
                                service=SERVICE, fit_started=False, peer_modified=False))
            return
        time.sleep(30)
    if not json.loads(FEATURE_COMPLETION.read_text())['complete']:
        raise ValueError('Incomplete feature receipt')
    # The older finite fitting waiter may have expired during overnight GPU use.
    # Starting an already active service is a no-op; otherwise renew only once.
    ensure_service(FIT_SERVICE, 'v3_fresh_scale_fit.py', '1800M', '2400M', 22200)
    write_progress(dict(stage='features_complete_fitting_service_started', service=FIT_SERVICE,
                        peer_modified=False, model_selected=False))


if __name__ == '__main__':
    main()
