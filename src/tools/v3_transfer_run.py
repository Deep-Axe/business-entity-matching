"""Finite serial local pilot queue with explicit failure and resource reporting."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from v3_transfer_data import ROOT, PILOT, write


def memory_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    raise RuntimeError('No RAM measurement')


def ready(relative):
    path = PILOT / relative
    return path.exists() and json.loads(path.read_text()).get('complete', False)


def run(resume=False, minimum_available_gib=2., stop_available_gib=.55, poll_seconds=10.):
    if not 0 < stop_available_gib < minimum_available_gib or not 1 <= poll_seconds <= 10:
        raise ValueError('Require positive ordered RAM guards and a 1–10 second poll')
    out = PILOT / 'queue'
    out.mkdir(exist_ok=resume)
    lock = (out / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resume and (out / 'progress.json').exists():
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        (out / f'progress_before_recovery_{stamp}.json').write_bytes((out / 'progress.json').read_bytes())
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'LD_LIBRARY_PATH'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_XET='1')
    cpu = str(ROOT / '.venv/bin/python')
    gpu = str(ROOT / '.venv-cascade-v2/bin/python')
    indic = str(ROOT / '.venv-indic-v3/bin/python')
    jobs = [
        ('e5_fit', [cpu, 'tools/v3_transfer_cpu.py', 'e5', '--extras', 'e5_features'], 'e5/report.json'),
        ('contrastive_encode', [gpu, 'tools/v3_transfer_gpu.py', 'contrastive'], 'contrastive_features/manifest.json'),
        ('contrastive_fit', [cpu, 'tools/v3_transfer_cpu.py', 'contrastive', '--extras', 'contrastive_features'], 'contrastive/report.json'),
        ('indic_romanize', [indic, 'tools/v3_transfer_indic.py', 'variants'], 'indic_features/romanization_manifest.json'),
        ('indic_features', [cpu, 'tools/v3_transfer_indic.py', 'features'], 'indic_features/manifest.json'),
        ('indic_fit', [cpu, 'tools/v3_transfer_cpu.py', 'indic', '--extras', 'indic_features'], 'indic/report.json'),
        ('xlmr_train_score', [gpu, 'tools/v3_transfer_gpu.py', 'xlmr'], 'xlmr/manifest.json'),
        ('xlmr_evaluate', [cpu, 'tools/v3_transfer_report.py', '--evaluate-xlmr'], 'xlmr/report.json'),
        ('report', [cpu, 'tools/v3_transfer_report.py'], 'summary.json'),
    ]
    state = dict(complete=False, stage='waiting_for_initial_e5', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), jobs=[],
                 resource_policy=dict(minimum_available_gib=minimum_available_gib,
                     stop_available_gib=stop_available_gib, poll_seconds=poll_seconds))
    deadline = time.monotonic()+4*3600
    while not ready('e5_features/manifest.json'):
        state.update(available_gib=memory_gib(), free_disk_gib=shutil.disk_usage(ROOT).free/1024**3)
        write(out / 'progress.json', state)
        if time.monotonic() > deadline:
            raise TimeoutError('Initial E5 did not complete within queue budget')
        time.sleep(15)
    for tag, command, marker in jobs:
        state['stage'] = tag
        if resume and ready(marker):
            entry = dict(tag=tag, complete=True, reused_saved_result=True)
            state['jobs'].append(entry)
            write(out / 'progress.json', state)
            print(json.dumps(entry), flush=True)
            continue
        if resume and tag == 'xlmr_train_score':
            command = command + ['--resume']
        while memory_gib() < minimum_available_gib or shutil.disk_usage(ROOT).free/1024**3 < 8:
            state.update(waiting_for_resources=True, available_gib=memory_gib(), free_disk_gib=shutil.disk_usage(ROOT).free/1024**3)
            write(out / 'progress.json', state)
            if time.monotonic() > deadline:
                raise TimeoutError('Resource wait exceeded finite queue budget')
            time.sleep(15)
        state['waiting_for_resources'] = False
        write(out / 'progress.json', state)
        start = time.monotonic()
        with (out / f'{tag}.log').open('a' if resume else 'w') as log:
            if resume:
                log.write('\n--- Resumed queue invocation ---\n')
                log.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            while child.poll() is None:
                state.update(available_gib=memory_gib(), free_disk_gib=shutil.disk_usage(ROOT).free/1024**3,
                             child_seconds=time.monotonic()-start, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                write(out / 'progress.json', state)
                if state['available_gib'] < stop_available_gib or state['free_disk_gib'] < 6 or time.monotonic() > deadline:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    state['resource_guard_stopped_child'] = True
                    break
                time.sleep(poll_seconds)
        entry = dict(tag=tag, exit_code=child.returncode, complete=child.returncode == 0 and ready(marker),
                     seconds=time.monotonic()-start)
        state['jobs'].append(entry)
        write(out / 'progress.json', state)
        print(json.dumps(entry), flush=True)
        if not entry['complete']:
            state['failed'] = tag
            write(out / 'progress.json', state)
            raise RuntimeError(f'{tag} failed; log {out / (tag + ".log")}; no automatic production changes')
    state.update(complete=True, stage='finished')
    write(out / 'progress.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--minimum-available-gib', type=float, default=2.)
    parser.add_argument('--stop-available-gib', type=float, default=.55)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    run(**vars(parser.parse_args()))
