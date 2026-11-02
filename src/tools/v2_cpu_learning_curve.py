"""Compare completed enhanced fits with nested cohorts and identical active recipes.

Model-size preference is recorded from tuning only. Development paired intervals
describe the learning curve; no audit or submitted predictions are accessed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ber.data import write_json
from ber.shards import file_hash
from v2_cascade_cache import validate_inputs
from v2_fit_cpu import ALL_NAMES, load, paired_bootstrap, score


def assert_array_prefix(old, new, label, batch=20000):
    """Bounded-memory check that size, not previously prepared examples, changed."""
    if old.ndim != new.ndim or old.shape[1:] != new.shape[1:] or len(old) > len(new):
        raise ValueError(f'Training array prefix dimensions differ: {label}')
    for start in range(0, len(old), batch):
        stop = min(start + batch, len(old))
        if not np.array_equal(old[start:stop], new[start:stop]):
            raise ValueError(f'Training array prefix values differ: {label}')
        for values in (old, new):
            if isinstance(values, np.memmap) and hasattr(values._mmap, 'madvise'):
                import mmap
                values._mmap.madvise(mmap.MADV_DONTNEED)


def run(runs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records, scores = [], {}
    expected_recipe = expected_fit = expected_tune = expected_dev = None
    previous_ids = previous_train = None
    ordered = sorted((Path(path) for path in runs), key=lambda path:
        json.loads((path / 'config.json').read_text())['training_references'])
    if len(set(ordered)) != len(ordered) or len(ordered) < 2:
        raise ValueError('Need at least two distinct completed model runs')
    for directory in ordered:
        config = json.loads((directory / 'config.json').read_text())
        report = json.loads((directory / 'enhanced_report.json').read_text())
        if report['features'] != ALL_NAMES or report['model_sha256'] != file_hash(directory / 'enhanced.cbm'):
            raise ValueError('Model schema/hash mismatch')
        train = Path(config['manifests']['train'])
        train_meta = json.loads((train / 'manifest.json').read_text())
        source = train_meta['source_config']
        recipe = {key: source[key] for key in ('gate_terms', 'gallery_fold', 'index_config_sha256',
            'lexicon_sha256', 'top_k_per_channel', 'channels', 'feature_code_sha256',
            'direct_feature_code_sha256', 'retrieval_code_sha256')}
        recipe['augmented_feature_code_sha256'] = train_meta['feature_code_sha256']
        fixed_fit = {key: config[key] for key in ('iterations', 'depth', 'learning_rate', 'random_seed')}
        if expected_recipe is None:
            expected_recipe, expected_fit = recipe, fixed_fit
        elif recipe != expected_recipe or fixed_fit != expected_fit:
            raise ValueError('Training recipe or fixed hyperparameters differ; not a controlled size curve')
        ids = np.load(train / 'reference_ids.npy', allow_pickle=False)
        count = len(ids)
        if count != config['training_references'] or count != train_meta['references']:
            raise ValueError('Training reference count mismatch')
        if previous_ids is not None and (count <= len(previous_ids) or not np.array_equal(ids[:len(previous_ids)], previous_ids)):
            raise ValueError('Training cohorts are not exact nested prefixes')
        if previous_train is not None:
            for name in ('X', 'y', 'query_ids', 'true_counts', 'countries'):
                assert_array_prefix(np.load(previous_train / f'{name}.npy', mmap_mode='r'),
                                    np.load(train / f'{name}.npy', mmap_mode='r'), name)
        previous_ids = ids
        previous_train = train
        card = dict(training_references=count, training_pairs=train_meta['pairs'],
            model_sha256=report['model_sha256'], threshold=report['threshold'],
            training_seconds=report['training_seconds'], peak_rss_mib=report['peak_rss_mib'],
            run=str(directory.resolve()), train_manifest_sha256=file_hash(train / 'manifest.json'))
        for split, matrix_key in (('full_tune', 'full_tune'), ('dev', 'dev')):
            prepared = Path(config['manifests'][matrix_key])
            arrays, matrix = load(prepared)
            digest = file_hash(prepared / 'manifest.json')
            if split == 'full_tune':
                if expected_tune is not None and digest != expected_tune:
                    raise ValueError('Tuning matrices differ')
                expected_tune = digest
            else:
                if expected_dev is not None and digest != expected_dev:
                    raise ValueError('Development matrices differ')
                expected_dev = digest
            _, _, binding, _, _, probabilities = validate_inputs(matrix['source'],
                directory / f'enhanced_{split}_probabilities.npy', prepared / 'manifest.json',
                store=matrix.get('resolved_store'), index=matrix.get('resolved_index'))
            if binding['model_sha256'] != report['model_sha256']:
                raise ValueError('Probability producer differs from the compared model')
            metrics, entity_scores = score(arrays, probabilities, report['threshold'])
            if abs(metrics['macro_f05'] - report[split]['macro_f05']) > 1e-12:
                raise ValueError('Saved probabilities do not reproduce reported metrics')
            card[split] = metrics
            if split == 'dev':
                scores[count] = entity_scores
        records.append(card)
    sizes = [card['training_references'] for card in records]
    comparisons = {f'{new}_vs_{old}': paired_bootstrap(scores[old], scores[new])
                   for old, new in zip(sizes[:-1], sizes[1:])}
    if len(sizes) > 2:
        comparisons[f'{sizes[-1]}_vs_{sizes[0]}'] = paired_bootstrap(scores[sizes[0]], scores[sizes[-1]])
    selected = max(records, key=lambda card: (card['full_tune']['macro_f05'], -card['training_references']))
    result = dict(models=records, paired_development_comparisons=comparisons,
        size_preferred_on_tuning_only=selected['training_references'],
        fixed_training_recipe=expected_recipe, fixed_hyperparameters=expected_fit,
        exact_nested_training_prefixes=True, exact_nested_training_arrays=True,
        full_tune_manifest_sha256=expected_tune,
        dev_manifest_sha256=expected_dev, audit_used=False, external_data_used=False,
        caveat='Conditional development intervals; no hidden-test or French accuracy claim. No submission changed.')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'report.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    run(**vars(parser.parse_args()))
