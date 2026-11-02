"""Prepare a train-only CatBoost quantized pool via a column-major disk memmap.

This is a separate process from fitting. All outputs are new; nothing is removed.
Numeric borders are learned only from a verified training-gallery matrix.
"""
import argparse
import gc
import json
import mmap
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from catboost import Pool

from ber.context_features import CONTEXT_NAMES
from ber.data import write_json
from ber.features import FEATURE_NAMES
from ber.shards import file_hash
from v2_cpu_features import EXTRA_NAMES

GIB = 1024**3


def available_gib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) / 1024**2


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('Cannot inspect resident memory')


def memory_requirement_gib(rows, columns, stage):
    floats = rows * columns * 4 / GIB
    bins = rows * columns / GIB  # CPU 254 borders fit in one byte per feature.
    if stage == 'quantize':
        return max(3.5, floats + 2 * bins + 1.5)
    if stage == 'quantized_fit':
        return max(3.5, 3 * bins + rows * 32 / GIB + 1.5)
    if stage == 'raw_fit':
        return max(3.5, 2 * floats + 1.5)
    raise ValueError(stage)


def release_pages(array):
    if isinstance(array, np.memmap):
        array._mmap.madvise(mmap.MADV_DONTNEED)


def copy_fortran(source, target, columns, batch=20000):
    destination = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32,
                                            shape=(len(source), columns), fortran_order=True)
    for start in range(0, len(source), batch):
        stop = min(start + batch, len(source))
        destination[start:stop] = source[start:stop, :columns]
        destination.flush()
        release_pages(destination)
        release_pages(source)
    if not destination.flags.f_contiguous:
        raise AssertionError('Expected column-major float32 data')
    del destination


def quantize(prepared, output, variant='enhanced', threads=2, minimum_available_gib=3.5):
    if not 1 <= threads <= 3:
        raise ValueError('Quantization is capped at three CPU cores')
    # Pool.quantize has no public thread_count option. Limit this process's CPU
    # affinity so native quantization workers cannot occupy the entire laptop.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:threads])
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['source_config']['gallery_fold'] != 'train':
        raise ValueError('Numeric borders may only be learned from a complete training-gallery matrix')
    names = FEATURE_NAMES + CONTEXT_NAMES + (EXTRA_NAMES if variant == 'enhanced' else [])
    if manifest['names'][:len(names)] != names:
        raise ValueError('Feature schema mismatch')
    rows, columns = manifest['pairs'], len(names)
    requirement = max(minimum_available_gib, memory_requirement_gib(rows, columns, 'quantize'))
    if available_gib() < requirement:
        raise RuntimeError(f'Quantization preflight requires {requirement:.2f} GiB available; have {available_gib():.2f}')
    disk_required = rows * columns * 5.5 + GIB
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < disk_required:
        raise RuntimeError('Insufficient disk for retained Fortran matrix, quantized pool and reserve')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = np.load(prepared / 'X.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(prepared / 'y.npy', mmap_mode='r', allow_pickle=False)
    if source.shape != (rows, len(manifest['names'])) or labels.shape != (rows,):
        raise ValueError('Prepared dimensions differ from manifest')
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Training pool needs both label classes')
    copy_fortran(source, output / 'X_fortran.npy', columns)
    del source
    gc.collect()
    data = np.load(output / 'X_fortran.npy', mmap_mode='r', allow_pickle=False)
    before_pool = rss_mib()
    pool = Pool(data, labels, feature_names=names, thread_count=threads)
    pool_delta = rss_mib() - before_pool
    # Guard the installed implementation's zero-copy behavior, not merely its
    # advertised type support. Mapped input pages are reclaimable and remain valid.
    if pool_delta > rows * columns * 4 / 1024**2 * .5 + 128:
        raise RuntimeError('Unexpected raw-pool memory copy; stop before quantization')
    release_pages(data)
    print(json.dumps(dict(stage='quantize_start', rows=rows, features=columns,
        available_gib=available_gib(), preflight_required_gib=requirement,
        pool_creation_rss_delta_mib=pool_delta)), flush=True)
    pool.quantize(border_count=254, feature_border_type='GreedyLogSum', random_seed=2026, task_type='CPU')
    pool.save(str(output / 'train.cbq'))
    pool.save_quantization_borders(str(output / 'train_borders.tsv'))
    report = dict(complete=True, variant=variant, names=names, pairs=rows,
        references=manifest['references'], borders_learned_from='training gallery only',
        prepared_manifest_sha256=file_hash(prepared / 'manifest.json'),
        input_matrix_sha256=file_hash(prepared / 'X.npy'), labels_sha256=file_hash(prepared / 'y.npy'),
        quantized_sha256=file_hash(output / 'train.cbq'), borders_sha256=file_hash(output / 'train_borders.tsv'),
        prepared=str(prepared), cpu_affinity=sorted(os.sched_getaffinity(0)),
        quantization=dict(border_count=254, feature_border_type='GreedyLogSum', random_seed=2026, task_type='CPU'),
        preflight_required_gib=requirement, pool_creation_rss_delta_mib=pool_delta,
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        quantized_bytes=(output / 'train.cbq').stat().st_size,
        used_ram_limit_is_hard_cap=False, external_data_used=False)
    write_json(output / 'manifest.json', report)
    print(json.dumps(report), flush=True)


def load_verified_pool(directory, prepared, names, threads):
    directory, prepared = Path(directory), Path(prepared)
    report = json.loads((directory / 'manifest.json').read_text())
    if not report['complete'] or report['names'] != names:
        raise ValueError('Quantized pool schema/incompleteness mismatch')
    for path, expected in ((prepared / 'manifest.json', report['prepared_manifest_sha256']),
                           (prepared / 'X.npy', report['input_matrix_sha256']),
                           (prepared / 'y.npy', report['labels_sha256']),
                           (directory / 'train.cbq', report['quantized_sha256']),
                           (directory / 'train_borders.tsv', report['borders_sha256'])):
        if file_hash(path) != expected:
            raise ValueError('Quantized pool/provenance changed')
    pool = Pool('quantized://' + str((directory / 'train.cbq').resolve()), thread_count=threads)
    if not pool.is_quantized() or pool.num_row() != report['pairs'] or pool.get_feature_names() != names:
        raise ValueError('Loaded quantized pool dimensions/names changed')
    return pool, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=('control', 'enhanced'), default='enhanced')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--minimum-available-gib', type=float, default=3.5)
    quantize(**vars(parser.parse_args()))
