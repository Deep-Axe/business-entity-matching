"""Stream existing feature shards to memmaps; controlled 46 vs 76 feature CPU fit.

Commands prepare / fit are separate so index mappings exit before CatBoost fit.
No frozen inputs are modified. Full-gallery tune chooses thresholds; dev measures
paired deltas only, and is never supplied to training or threshold selection.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
from catboost import CatBoostClassifier, Pool

from ber.context_features import CONTEXT_NAMES, context_matrix
from ber.data import read_tsv, write_json
from ber.features import FEATURE_NAMES
from ber.metrics import score_arrays, tune_threshold
from ber.shards import file_hash
from ber.store import connect, lookup, record_row
from v2_cpu_features import EXTRA_NAMES, View, Weighting, extra_features

BASE_NAMES = FEATURE_NAMES + CONTEXT_NAMES
ALL_NAMES = BASE_NAMES + EXTRA_NAMES


def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('Cannot determine free RAM')


def cache_inventory(source):
    source = Path(source).resolve()
    config = json.loads((source / 'config.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Source cache incomplete')
    shards = []
    for path in sorted((source / 'shards').glob('[0-9]*')):
        done = json.loads((path / 'done.json').read_text())
        shards.append((path, done))
    return source, config, shards


def prepare(source, output, store=None, index=None):
    source, config, shards = cache_inventory(source)
    resolved_store = Path(store or config['store']).resolve()
    resolved_index = Path(index or config['index']).resolve()
    if file_hash(resolved_index / 'config.json') != config['index_config_sha256']:
        raise ValueError('Resolved index differs from the original cache index recipe')
    index_config = json.loads((resolved_index / 'config.json').read_text())
    if file_hash(resolved_store / 'config.json') != index_config['store_config_sha256']:
        raise ValueError('Resolved record store differs from the indexed record store')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    total = sum(done['candidate_pairs'] for _, done in shards)
    references = sum(done['references'] for _, done in shards)
    matrix = np.lib.format.open_memmap(output / 'X.npy', mode='w+', dtype=np.float32, shape=(total, len(ALL_NAMES)))
    labels = np.lib.format.open_memmap(output / 'y.npy', mode='w+', dtype=np.int8, shape=(total,))
    query_ids = np.lib.format.open_memmap(output / 'query_ids.npy', mode='w+', dtype=np.int32, shape=(total,))
    counts = np.lib.format.open_memmap(output / 'true_counts.npy', mode='w+', dtype=np.int32, shape=(references,))
    db = connect(resolved_store)
    weights = Weighting(resolved_index)
    offset = qoffset = 0
    started = time.perf_counter()
    reference_ids, countries, index = [], [], []
    for shard_number, (shard, done) in enumerate(shards):
        if file_hash(shard / 'features.npz') != done['output_sha256']['features.npz']:
            raise ValueError('Source features changed')
        with np.load(shard / 'features.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in ('X', 'y', 'query_ids', 'reference_ids', 'target_ids', 'true_counts', 'countries')}
        n, nq = len(arrays['y']), len(arrays['reference_ids'])
        if n != done['candidate_pairs'] or nq != done['references']:
            raise ValueError('Shard dimensions changed')
        rows = {r['entity_id']: r for r in read_tsv(shard / 'references.tsv')}
        anchors = [View.from_row(rows[str(eid)]) for eid in arrays['reference_ids']]
        raw_targets = lookup(db, 'records', 'entity_id', list(set(map(str, arrays['target_ids']))))
        target_views = {eid: View.from_row(record_row(row)) for eid, row in raw_targets.items()}
        del raw_targets
        matrix[offset:offset + n, :len(BASE_NAMES)] = context_matrix(arrays)
        extra = np.asarray([extra_features(anchors[q], target_views[str(target)], weights)
                            for q, target in zip(arrays['query_ids'], arrays['target_ids'])], dtype=np.float32)
        matrix[offset:offset + n, len(BASE_NAMES):] = extra
        labels[offset:offset + n] = arrays['y']
        query_ids[offset:offset + n] = arrays['query_ids'] + qoffset
        counts[qoffset:qoffset + nq] = arrays['true_counts']
        reference_ids.extend(map(str, arrays['reference_ids']))
        countries.extend(map(str, arrays['countries']))
        index.append(dict(shard=shard.name, pair_start=offset, pair_end=offset + n,
                          query_start=qoffset, query_end=qoffset + nq,
                          input_sha256=done['input_sha256'], features_sha256=done['output_sha256']['features.npz']))
        offset += n
        qoffset += nq
        # Release dirty memmap pages while retaining file-backed data on disk.
        for mapping in (matrix, labels, query_ids, counts):
            mapping.flush()
            if hasattr(mapping._mmap, 'madvise'):
                import mmap
                mapping._mmap.madvise(mmap.MADV_DONTNEED)
        if shard_number % 5 == 0 or shard_number == len(shards) - 1:
            progress = dict(stage='prepare', source=source.name, shards=shard_number + 1, total_shards=len(shards),
                pairs=offset, seconds=time.perf_counter() - started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                available_gib=available_gib())
            write_json(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    assert offset == total and qoffset == references
    if len(set(reference_ids)) != references:
        raise ValueError('Duplicate references')
    np.save(output / 'reference_ids.npy', np.asarray(reference_ids))
    np.save(output / 'countries.npy', np.asarray(countries))
    report = dict(complete=True, source=str(source), source_config=config,
        resolved_store=str(resolved_store), resolved_index=str(resolved_index),
        original_cache_metadata_modified=False,
        source_config_sha256=file_hash(source / 'config.json'), names=ALL_NAMES,
        feature_code_sha256=file_hash(Path(__file__).with_name('v2_cpu_features.py')),
        runner_sha256=file_hash(__file__), references=references, pairs=total,
        shards=index, seconds=time.perf_counter() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        label_free_features=True, external_data_used=False)
    write_json(output / 'manifest.json', report)
    db.close()


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['names'] != ALL_NAMES:
        raise ValueError('Incomplete or inconsistent feature matrix')
    arrays = {key: np.load(directory / f'{key}.npy', mmap_mode='r', allow_pickle=False)
              for key in ('X', 'y', 'query_ids', 'true_counts', 'reference_ids', 'countries')}
    return arrays, manifest


def score(arrays, probability, threshold):
    result = score_arrays(arrays['query_ids'], arrays['y'], probability, arrays['true_counts'], threshold)
    scores = result.pop('entity_scores')
    result['by_country'] = {str(country): dict(references=int((arrays['countries'] == country).sum()),
        macro_f05=float(scores[arrays['countries'] == country].mean())) for country in np.unique(arrays['countries'])}
    return result, scores


def paired_bootstrap(old, new, repetitions=3000):
    delta = np.asarray(new) - np.asarray(old)
    rng = np.random.default_rng(20260925)
    estimates = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repetitions)])
    return dict(delta=float(delta.mean()), ci95=list(map(float, np.quantile(estimates, [.025, .975]))),
                improved_references=int((delta > 0).sum()), worsened_references=int((delta < 0).sum()),
                repetitions=repetitions, conditional_on_selected_model=True)


def validate_recipes(manifests):
    configs = {name: value['source_config'] for name, value in manifests.items()}
    # The tune-fold matrix is an identity/separation registry only. None of its
    # features or scores train the model or select the final threshold.
    active = [configs[name] for name in ('train', 'full_tune', 'dev')]
    for key in ('gate_terms', 'lexicon_sha256', 'top_k_per_channel', 'index_config_sha256'):
        if len({json.dumps(config[key], sort_keys=True) for config in active}) != 1:
            raise ValueError(f'Mismatched active cache recipe: {key}')
    if configs['tune']['index_config_sha256'] != configs['train']['index_config_sha256']:
        raise ValueError('Identity-only tuning registry belongs to a different index')
    if configs['train']['gallery_fold'] != 'train' or configs['tune']['gallery_fold'] != 'tune':
        raise ValueError('Training and tuning registry must use their disjoint owner galleries')
    if configs['full_tune']['gallery_fold'] is not None or configs['dev']['gallery_fold'] is not None:
        raise ValueError('full-tune and full-dev must retrieve held-out owners against the complete training gallery')
    if not configs['full_tune'].get('labels_available') or not configs['dev'].get('labels_available'):
        raise ValueError('full-tune and full-dev require labeled full-gallery features')


def fit(train, tune, full_tune, full_dev, output, threads=3, iterations=600,
        minimum_available_gib=3.5, frozen_model=None,
        quantized_train_directory=None, variants=('control', 'enhanced')):
    from v2_cpu_quantized import load_verified_pool, memory_requirement_gib
    if len(set(variants)) != len(variants) or not set(variants) <= {'control', 'enhanced'} or not variants:
        raise ValueError('Choose unique control/enhanced variants')
    if not 1 <= threads <= 3:
        raise ValueError('This local controlled pilot is capped at three CPU threads')
    if available_gib() < minimum_available_gib:
        raise RuntimeError(f'Insufficient available RAM: {available_gib():.2f} GiB')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    datasets, manifests = {}, {}
    for name, directory in (('train', train), ('tune', tune), ('full_tune', full_tune), ('dev', full_dev)):
        datasets[name], manifests[name] = load(directory)
    validate_recipes(manifests)
    sets = {name: set(data['reference_ids']) for name, data in datasets.items()}
    if sets['train'] & (sets['tune'] | sets['dev']) or sets['tune'] & sets['dev'] or sets['tune'] != sets['full_tune']:
        raise ValueError('Reference overlap or mismatched calibration cohort')
    config = dict(threads=threads, iterations=iterations, depth=8, learning_rate=.07, random_seed=2026,
        variants=list(variants), quantized_train_directory=quantized_train_directory,
        tune_fold_role='cohort identity/separation only; no feature use or comparable tune-fold scores',
        training_references=len(sets['train']), manifests={name: str(Path(path).resolve()) for name, path in
            (('train', train), ('tune', tune), ('full_tune', full_tune), ('dev', full_dev))},
        external_data_used=False, audit_used=False, threshold_selection='full_tune only')
    write_json(output / 'config.json', config)
    dev = datasets['dev']
    frozen_scores = None
    reports = {}
    if frozen_model:
        frozen_path = Path(frozen_model)
        if not frozen_path.exists():
            raise FileNotFoundError(frozen_path)
        baseline = CatBoostClassifier(thread_count=threads)
        baseline.load_model(str(frozen_path))
        nfeat = len(baseline.feature_names_)
        if nfeat > dev['X'].shape[1]:
            raise ValueError(f'Comparison model has {nfeat} features; prepared matrix has {dev["X"].shape[1]}')
        frozen_probability = baseline.predict_proba(dev['X'][:, :nfeat], thread_count=threads)[:, 1]
        frozen_report, frozen_scores = score(dev, frozen_probability, .74)
        np.save(output / 'frozen_dev_probabilities.npy', frozen_probability)
        reports['frozen'] = dict(threshold=.74, dev=frozen_report, model_sha256=file_hash(frozen_path),
                                 path=str(frozen_path), features=int(nfeat),
                                 note='optional historical comparison; unused by the published cascade')
        del baseline
    else:
        reports['frozen'] = dict(skipped=True,
                                 reason='historical baseline comparison is optional and unused by the published cascade')
    trained_scores = {}
    for variant, names in (('control', BASE_NAMES), ('enhanced', ALL_NAMES)):
        if variant not in variants:
            continue
        stage = 'quantized_fit' if quantized_train_directory else 'raw_fit'
        requirement = max(minimum_available_gib,
            memory_requirement_gib(len(datasets['train']['y']), len(names), stage))
        if available_gib() < requirement:
            raise RuntimeError(f'Insufficient RAM before {variant}: need {requirement:.2f}, have {available_gib():.2f} GiB')
        stamp = time.perf_counter()
        model = CatBoostClassifier(iterations=iterations, depth=8, learning_rate=.07,
            loss_function='Logloss', random_seed=2026, thread_count=threads,
            allow_writing_files=False, verbose=100, used_ram_limit='2500mb')
        print(json.dumps(dict(stage='fit_start', variant=variant, available_gib=available_gib())), flush=True)
        quantized_provenance = None
        if quantized_train_directory:
            pool, quantized_provenance = load_verified_pool(
                Path(quantized_train_directory) / variant, train, names, threads)
        else:
            pool = Pool(datasets['train']['X'][:, :len(names)], datasets['train']['y'], feature_names=names)
        model.fit(pool)
        model.save_model(str(output / f'{variant}.cbm'))
        train_seconds = time.perf_counter() - stamp
        del pool
        gc.collect()
        ft = datasets['full_tune']
        probabilities = model.predict_proba(ft['X'][:, :len(names)], thread_count=threads)[:, 1]
        threshold, _ = tune_threshold(ft['query_ids'], ft['y'], probabilities, ft['true_counts'])
        tune_report, _ = score(ft, probabilities, threshold)
        np.save(output / f'{variant}_full_tune_probabilities.npy', probabilities)
        probabilities = model.predict_proba(dev['X'][:, :len(names)], thread_count=threads)[:, 1]
        dev_report, scores = score(dev, probabilities, threshold)
        trained_scores[variant] = scores
        np.save(output / f'{variant}_dev_probabilities.npy', probabilities)
        report = dict(threshold=threshold, full_tune=tune_report, dev=dev_report,
            training_pool=stage, memory_preflight_gib=requirement,
            quantized_provenance=quantized_provenance,
            paired_delta_vs_frozen=paired_bootstrap(frozen_scores, scores) if frozen_scores is not None else None,
            training_seconds=train_seconds,
            features=names, feature_importance=sorted(zip(names, map(float, model.feature_importances_)), key=lambda p: -p[1]),
            model_sha256=file_hash(output / f'{variant}.cbm'),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        reports[variant] = report
        write_json(output / f'{variant}_report.json', report)
        print(json.dumps(dict(stage='fit_complete', variant=variant, training_seconds=train_seconds,
            threshold=threshold, dev=dev_report, delta=report['paired_delta_vs_frozen'])), flush=True)
        del model
        gc.collect()
    if {'control', 'enhanced'} <= set(trained_scores):
        reports['enhanced_vs_control'] = paired_bootstrap(trained_scores['control'], trained_scores['enhanced'])
    write_json(output / 'report.json', reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--store', help='Local equivalent of a cache record-store path; checksum guarded')
    prep.add_argument('--index', help='Local equivalent of a cache index path; checksum guarded')
    fitting = commands.add_parser('fit')
    for name in ('train', 'tune', 'full-tune', 'full-dev', 'output'):
        fitting.add_argument('--' + name, required=True)
    fitting.add_argument('--threads', type=int, default=3)
    fitting.add_argument('--iterations', type=int, default=600)
    fitting.add_argument('--minimum-available-gib', type=float, default=3.5)
    fitting.add_argument('--frozen-model', default=None,
                        help='optional historical CatBoost for a paired-delta report')
    fitting.add_argument('--quantized-train-directory', help='Root containing checksum-bound control/enhanced quantized pool folders')
    fitting.add_argument('--variants', nargs='+', choices=('control', 'enhanced'), default=['control', 'enhanced'])
    args = vars(parser.parse_args())
    command = args.pop('command')
    (prepare if command == 'prepare' else fit)(**args)
