"""Recover the single interrupted local inference assignment after a reboot.

Never touches EC2 jobs, predictions, model/runner code, or frozen queue config.
Requires the controller lock, proves the old process is dead, verifies committed
shard checksums, archives original job metadata, and adds only --resume.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from ber.shards import file_hash
from v2_remote_job import validate_spec, write_json
from v2_submission_controller import ROOT


def live(state):
    path = Path(f'/proc/{state.get("pid")}/stat')
    if not path.exists():
        return False
    fields = path.read_text().split()
    return len(fields) > 21 and fields[21] == state.get('start_ticks') and fields[2] != 'Z'


def run():
    controller = ROOT / 'artifacts/improvement_v2/inference/submission_controller_v1'
    lock = (controller / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg_path = ROOT / 'artifacts/improvement_v2/inference/submission_queue_config_v1.json'
    cfg = json.loads(cfg_path.read_text())
    queue = json.loads((controller / 'queue.json').read_text())
    if queue['config_sha256'] != file_hash(cfg_path):
        raise ValueError('Frozen controller configuration changed')
    part = queue['assigned'].get('local')
    if part is None:
        print(json.dumps(dict(restart_needed=False, reason='No active local assignment')))
        return
    item = queue['specs'][str(part)]
    validate_spec(item)
    jobs = ROOT / 'artifacts/v2_remote_jobs'
    state_path = jobs / (item['job'] + '.json')
    exit_path = jobs / (item['job'] + '.exit.json')
    state = json.loads(state_path.read_text())
    if live(state):
        raise RuntimeError('Existing local job is alive; do not duplicate it')
    if exit_path.exists():
        raise RuntimeError('Terminal exit exists; inspect failure instead of blindly restarting')
    if state['output'] != item['output'] or [v for v in state['argv'] if v != '--resume'] != item['argv']:
        raise ValueError('Job argv/output differs from frozen queue assignment')
    directory = ROOT / item['output']
    recipe_path = directory / 'config.json'
    recipe = json.loads(recipe_path.read_text())
    if (recipe['partition'] != part or recipe['partitions'] != cfg['partitions']
            or recipe['model_sha256'] != cfg['model_sha256']
            or file_hash(ROOT / 'tools/v2_run_pipeline.py') != cfg['runner_sha256']
            or file_hash(recipe['model']) != cfg['model_sha256']):
        raise ValueError('Prediction/model/runner identity mismatch')
    config_sha = file_hash(recipe_path)
    saved = []
    for shard in sorted((directory / 'shards').iterdir()):
        if not shard.name.isdigit():
            continue  # Preserve uncommitted temp directories; runner ignores them.
        d = json.loads((shard / 'done.json').read_text())
        if (int(shard.name) % cfg['partitions'] != part or d['shard_id'] != int(shard.name)
                or d['config_sha256'] != config_sha):
            raise ValueError('Saved shard identity mismatch')
        for name, digest in d['output_sha256'].items():
            if Path(name).name != name or file_hash(shard / name) != digest:
                raise ValueError('Saved shard checksum mismatch')
        saved.append(dict(shard=d['shard_id'], references=d['references'], done_sha256=file_hash(shard / 'done.json')))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = ROOT / 'artifacts/recovery' / stamp
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(state_path, archive / state_path.name)
    shutil.copy2(controller / 'queue.json', archive / 'queue100k_before.json')
    shutil.copy2(recipe_path, archive / 'local_recipe_before.json')
    state['argv'] = item['argv'] + ['--resume']
    state['recovered_utc'] = datetime.now(timezone.utc).isoformat()
    state['recovery_archive'] = str(archive)
    state['recovery_boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    write_json(state_path, state)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (jobs / (item['job'] + '.log')).open('a') as log:
        log.write('\nRECOVERY ' + state['recovered_utc'] + ': resume verified committed shards after reboot\n')
        log.flush()
        child = subprocess.Popen(['.venv/bin/python', 'tools/v2_remote_job.py', 'execute', '--job', item['job']],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state.update(pid=child.pid, start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])
    write_json(state_path, state)
    receipt = dict(restart_launched=True, job=item['job'], partition=part, pid=child.pid,
        saved_shards=len(saved), saved_references=sum(s['references'] for s in saved), shards=saved,
        config_sha256=config_sha, resume_only_argument_added=True, predictions_deleted=False,
        controller_config_sha256=queue['config_sha256'], original_state=str(archive / state_path.name))
    write_json(archive / 'local_inference_recovery.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'shards'}), flush=True)


if __name__ == '__main__':
    run()
