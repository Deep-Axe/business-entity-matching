"""Finite release-bound watcher that scores a completed fresh audit exactly once."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ber.shards import file_hash


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run(source, references, release, output, watcher, pipeline_pid, minutes=60, interval=20):
    source, references, release, output, watcher = map(Path, (source, references, release, output, watcher))
    if not 0 < minutes <= 120 or not 1 <= interval <= 60:
        raise ValueError('Expected a finite bounded watcher interval/window')
    watcher.mkdir(parents=True, exist_ok=True)
    lock = (watcher / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scorer = Path(__file__).with_name('v2_assemble_submission.py')
    integrity = Path(__file__).with_name('v2_prediction_integrity.py')
    bindings = {str(path.resolve()): file_hash(path) for path in
                (release, references, source / 'config.json', scorer, integrity)}
    config = read(source / 'config.json')
    frozen = read(release)
    if (config.get('reference_count') != 10000 or not config.get('labels_available')
            or config.get('threshold') != frozen['threshold']
            or config.get('model_sha256') != frozen['model_sha256']
            or frozen['owner_policy']['enabled']):
        raise ValueError('Expected exact CPU-only fresh audit configuration')
    stat = Path(f'/proc/{pipeline_pid}/stat')
    fields = stat.read_text().split() if stat.exists() else []
    start_ticks = fields[21] if fields else None
    if fields:
        command = Path(f'/proc/{pipeline_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'tools/v2_run_pipeline.py' not in command or str(source) not in command:
            raise ValueError('Pipeline PID does not identify the owned audit job')

    def update(stage, **extra):
        value = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(),
                     release_sha256=bindings[str(release.resolve())], **extra)
        pending = watcher / 'progress.pending.json'
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(watcher / 'progress.json')
        print(json.dumps(value), flush=True)

    deadline = time.monotonic() + minutes * 60
    try:
        while time.monotonic() < deadline:
            for path, digest in bindings.items():
                if file_hash(path) != digest:
                    raise ValueError('Frozen watcher input changed: ' + path)
            if read(source / 'manifest.json').get('complete'):
                break
            fields = stat.read_text().split() if stat.exists() else []
            if not fields or fields[21] != start_ticks or fields[2] == 'Z':
                raise RuntimeError('Owned audit pipeline exited before complete manifest')
            progress = read(source / 'progress.json')
            update('waiting_for_complete_audit', references=progress.get('references', 0),
                   expected_references=10000)
            time.sleep(interval)
        else:
            raise TimeoutError('Finite audit watcher window exhausted; no score or ready claim')
        if not (output / 'report.json').exists():
            update('validating_and_scoring_frozen_policy')
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            env.pop('PYTHONPATH', None)
            subprocess.run([sys.executable, str(scorer), 'audit', '--sources', str(source),
                '--references', str(references), '--release', str(release), '--output', str(output)],
                check=True, env=env, timeout=600)
        report = read(output / 'report.json')
        if (not report.get('complete') or report.get('references') != 10000
                or report.get('release_sha256') != bindings[str(release.resolve())]
                or report.get('policy_tuned_on_audit') is not False):
            raise ValueError('Final audit report is incomplete or bound to another policy')
        update('complete', complete=True, report=str((output / 'report.json').resolve()),
               report_sha256=file_hash(output / 'report.json'))
    except Exception as error:
        update('attention_required', complete=False, error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'references', 'release', 'output', 'watcher'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--pipeline-pid', type=int, required=True)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--interval', type=float, default=20)
    run(**vars(parser.parse_args()))
