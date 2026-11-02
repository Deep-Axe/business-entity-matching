"""Separate, resource-guarded 200k CPU candidate; no audit or submission changes.

Large prepared/quantized training matrices remain on the dedicated cloud node.
A compact exact-prefix fingerprint proves compatibility with the local 100k fit.
Existing CPU preparation, quantization and fitting implementations are reused.
"""
import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ARRAYS = ('X', 'y', 'query_ids', 'true_counts', 'countries', 'reference_ids')
PAIR_ARRAYS = {'X', 'y', 'query_ids'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


def array_fingerprint(path, rows, batch=20000):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if rows < 1 or rows > len(array) or array.dtype.hasobject:
        raise ValueError('Invalid prefix length or unsupported object dtype')
    h = hashlib.sha256()
    for start in range(0, rows, batch):
        h.update(array[start:min(start + batch, rows)].tobytes(order='C'))
        if hasattr(array._mmap, 'madvise'):
            array._mmap.madvise(mmap.MADV_DONTNEED)
    return dict(rows=rows, shape=[rows, *array.shape[1:]], dtype=array.dtype.str, sha256=h.hexdigest())


def expectation(prepared, output):
    prepared = Path(prepared)
    meta = json.loads((prepared / 'manifest.json').read_text())
    if not meta['complete'] or meta['source_config']['gallery_fold'] != 'train':
        raise ValueError('Expected a complete training-only matrix')
    value = dict(references=meta['references'], pairs=meta['pairs'], names=meta['names'],
        feature_code_sha256=meta['feature_code_sha256'],
        prepared_manifest_sha256=digest(prepared / 'manifest.json'), arrays={})
    for name in ARRAYS:
        rows = meta['pairs'] if name in PAIR_ARRAYS else meta['references']
        value['arrays'][name] = array_fingerprint(prepared / (name + '.npy'), rows)
    write_new(output, value)
    return value


def verify_prefix(prepared, expected):
    prepared = Path(prepared)
    meta = json.loads((prepared / 'manifest.json').read_text())
    if (not meta['complete'] or meta['names'] != expected['names']
            or meta['feature_code_sha256'] != expected['feature_code_sha256']
            or meta['references'] <= expected['references']):
        raise ValueError('Training schema, code, completeness or nested-size mismatch')
    for name in ARRAYS:
        want = expected['arrays'][name]
        got = array_fingerprint(prepared / (name + '.npy'), want['rows'])
        if got != want:
            raise ValueError('Exact prepared training prefix differs: ' + name)
    return dict(exact_array_prefix=True, references=expected['references'], pairs=expected['pairs'],
        arrays_verified=list(ARRAYS), expected_manifest_sha256=expected['prepared_manifest_sha256'])


def available_gib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) / 1024 ** 2


def train(source, prepared, quantized, fit, tune, full_tune, full_dev,
          prefix_expectation, receipts, store='artifacts/train_store_v2',
          index='artifacts/train_index_v2'):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    source = Path(source)
    meta = json.loads((source / 'manifest.json').read_text())
    if (not meta['complete'] or meta['references'] != 200000
            or meta['config']['gallery_fold'] != 'train' or meta['config']['gate_terms'] != 0):
        raise ValueError('Expected verified 200k gate-0 training-gallery source')
    for name in (prepared, quantized, fit, receipts):
        path = Path(name).resolve()
        if root not in path.parents or path.exists():
            raise ValueError('Outputs must be new directories within the project')
    receipts = Path(receipts)
    receipts.mkdir(parents=True, exist_ok=False)
    required_disk = meta['candidate_pairs'] * (76 * 9.5 + 5) + 2 * 1024 ** 3
    if shutil.disk_usage(root).free < required_disk:
        raise RuntimeError('Insufficient disk for retained prepared/Fortran/quantized matrices plus reserve')
    expected = json.loads(Path(prefix_expectation).read_text())
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    python = str(root / '.venv/bin/python')
    stages = [
        ('prepare', [python, 'tools/v2_fit_cpu.py', 'prepare', '--source', str(source),
            '--output', prepared, '--store', store, '--index', index]),
        ('quantize', [python, 'tools/v2_cpu_quantized.py', '--prepared', prepared,
            '--output', str(Path(quantized) / 'enhanced'), '--variant', 'enhanced', '--threads', '2']),
        ('fit', [python, 'tools/v2_fit_cpu.py', 'fit', '--train', prepared, '--tune', tune,
            '--full-tune', full_tune, '--full-dev', full_dev, '--output', fit,
            '--threads', '2', '--iterations', '600', '--quantized-train-directory', quantized,
            '--variants', 'enhanced'])]
    started = time.time()
    try:
        for stage, argv in stages:
            stamp = time.time()
            write_new(receipts / (stage + '_start.json'), dict(utc_epoch=stamp, available_gib=available_gib(),
                disk_free_bytes=shutil.disk_usage(root).free, argv=argv))
            print(json.dumps(dict(stage=stage, state='starting')), flush=True)
            with (receipts / (stage + '.log')).open('x') as log:
                subprocess.run(argv, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            write_new(receipts / (stage + '_complete.json'), dict(success=True, seconds=time.time() - stamp))
            if stage == 'prepare':
                write_new(receipts / 'prefix_verification.json', verify_prefix(prepared, expected))
        # Bind predictions at their producer using byte-identical uploaded matrices.
        from v2_cascade_cache import write_probability_binding
        for split, matrix in (('full_tune', full_tune), ('dev', full_dev)):
            write_probability_binding(Path(fit) / ('enhanced_' + split + '_probabilities.npy'),
                Path(matrix) / 'manifest.json', Path(fit) / 'enhanced.cbm')
        write_new(receipts / 'complete.json', dict(complete=True, seconds=time.time() - started,
            source_manifest_sha256=digest(source / 'manifest.json'), audit_used=False,
            submission_changed=False, large_matrices_retained_on_cloud=True,
            model_sha256=digest(Path(fit) / 'enhanced.cbm')))
    except Exception as error:
        write_new(receipts / 'failure.json', dict(complete=False, error_type=type(error).__name__,
            message=str(error), seconds=time.time() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    exp = commands.add_parser('expectation')
    exp.add_argument('--prepared', required=True)
    exp.add_argument('--output', required=True)
    run = commands.add_parser('train')
    for name in ('source', 'prepared', 'quantized', 'fit', 'tune', 'full-tune', 'full-dev',
                 'prefix-expectation', 'receipts'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--store', default='artifacts/train_store_v2')
    run.add_argument('--index', default='artifacts/train_index_v2')
    args = vars(parser.parse_args())
    command = args.pop('command')
    (expectation if command == 'expectation' else train)(**args)
