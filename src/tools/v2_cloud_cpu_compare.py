"""Verify a separately trained cloud model against matched local CPU baselines.

Large cloud training matrices need not be copied back. Their nested-prefix proof,
prepared/quantized provenance, and exact saved-model evaluation replay are checked.
Only the existing tuning/development cohorts are used; no audit is accepted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ber.shards import file_hash
from v2_cascade_cache import validate_inputs
from v2_cloud_cpu_train import ARRAYS, write_new
from v2_fit_cpu import ALL_NAMES, load, paired_bootstrap, score


def run(cloud_fit, cloud_prepared_manifest, cloud_quantized_manifest, receipts,
        prefix_expectation, source, baseline_runs, full_tune, full_dev, output):
    cloud_fit, receipts, output = map(Path, (cloud_fit, receipts, output))
    if output.exists():
        raise FileExistsError(output)
    prepared = json.loads(Path(cloud_prepared_manifest).read_text())
    quantized = json.loads(Path(cloud_quantized_manifest).read_text())
    expected = json.loads(Path(prefix_expectation).read_text())
    proof = json.loads((receipts / 'prefix_verification.json').read_text())
    completion = json.loads((receipts / 'complete.json').read_text())
    source_meta = json.loads((Path(source) / 'manifest.json').read_text())
    if (not prepared['complete'] or prepared['references'] != 200000
            or not source_meta['complete'] or source_meta['references'] != 200000
            or prepared['source_config']['gallery_fold'] != 'train'
            or prepared['source_config']['gate_terms'] != 0
            or prepared['source_config_sha256'] != file_hash(Path(source) / 'config.json')
            or completion['source_manifest_sha256'] != file_hash(Path(source) / 'manifest.json')
            or not completion['complete'] or completion['audit_used']
            or not quantized['complete']
            or quantized['prepared_manifest_sha256'] != file_hash(cloud_prepared_manifest)):
        raise ValueError('Incomplete or inconsistent 200k training provenance')
    if (not proof['exact_array_prefix'] or proof['arrays_verified'] != list(ARRAYS)
            or proof['expected_manifest_sha256'] != expected['prepared_manifest_sha256']
            or proof['references'] != expected['references'] or proof['pairs'] != expected['pairs']
            or prepared['feature_code_sha256'] != expected['feature_code_sha256']):
        raise ValueError('Nested training-array proof differs from expected 100k parent')
    model = CatBoostClassifier(thread_count=1)
    model.load_model(str(cloud_fit / 'enhanced.cbm'))
    model_hash = file_hash(cloud_fit / 'enhanced.cbm')
    if completion['model_sha256'] != model_hash or list(model.feature_names_) != ALL_NAMES:
        raise ValueError('Collected model hash/schema mismatch')
    cloud_config = json.loads((cloud_fit / 'config.json').read_text())
    fixed_keys = ('iterations', 'depth', 'learning_rate', 'random_seed')
    recipe_keys = ('gate_terms', 'gallery_fold', 'index_config_sha256', 'lexicon_sha256',
                   'top_k_per_channel', 'channels', 'feature_code_sha256',
                   'direct_feature_code_sha256', 'retrieval_code_sha256')
    records, dev_scores = [], {}
    found_parent = False
    for directory in [*map(Path, baseline_runs), cloud_fit]:
        config = json.loads((directory / 'config.json').read_text())
        report = json.loads((directory / 'enhanced_report.json').read_text())
        n = config['training_references']
        train_meta = prepared if directory == cloud_fit else json.loads(
            (Path(config['manifests']['train']) / 'manifest.json').read_text())
        if n in dev_scores or n != train_meta['references'] or report['features'] != ALL_NAMES:
            raise ValueError('Duplicate training size or model schema/count mismatch')
        if any(config[key] != cloud_config[key] for key in fixed_keys):
            raise ValueError('Unmatched training hyperparameters')
        if any(train_meta['source_config'][key] != prepared['source_config'][key] for key in recipe_keys):
            raise ValueError('Unmatched retrieval/training recipe')
        if train_meta['feature_code_sha256'] != prepared['feature_code_sha256']:
            raise ValueError('Augmented feature code mismatch')
        if n == expected['references']:
            found_parent = file_hash(Path(config['manifests']['train']) / 'manifest.json') == expected['prepared_manifest_sha256']
        digest = file_hash(directory / 'enhanced.cbm')
        if digest != report['model_sha256']:
            raise ValueError('Model/report digest mismatch')
        record = dict(training_references=n, training_pairs=train_meta['pairs'],
            threshold=report['threshold'], model_sha256=digest,
            training_seconds=report['training_seconds'], peak_rss_mib=report['peak_rss_mib'])
        for split, matrix_dir in (('full_tune', full_tune), ('dev', full_dev)):
            arrays, matrix = load(matrix_dir)
            _, _, binding, _, _, probabilities = validate_inputs(matrix['source'],
                directory / ('enhanced_' + split + '_probabilities.npy'), Path(matrix_dir) / 'manifest.json',
                store=matrix.get('resolved_store'), index=matrix.get('resolved_index'))
            if binding['model_sha256'] != digest:
                raise ValueError('Probability/model provenance mismatch')
            if directory == cloud_fit:
                replay = model.predict_proba(arrays['X'], thread_count=1)[:, 1]
                if not np.array_equal(replay, probabilities):
                    raise ValueError('Saved cloud model does not exactly reproduce ' + split)
            metrics, values = score(arrays, probabilities, report['threshold'])
            if abs(metrics['macro_f05'] - report[split]['macro_f05']) > 1e-12:
                raise ValueError('Saved probabilities do not reproduce the report')
            record[split] = metrics
            if split == 'dev':
                dev_scores[n] = values
        records.append(record)
    if not found_parent or 200000 not in dev_scores:
        raise ValueError('Missing verified original 100k parent or 200k candidate')
    records.sort(key=lambda item: item['training_references'])
    comparisons = {f'200000_vs_{n}': paired_bootstrap(values, dev_scores[200000])
                   for n, values in dev_scores.items() if n != 200000}
    selected = max(records, key=lambda item: (item['full_tune']['macro_f05'], -item['training_references']))
    result = dict(complete=True, models=records, paired_development_comparisons=comparisons,
        size_preferred_on_tuning_only=selected['training_references'],
        exact_100k_training_array_prefix_verified_on_cloud=True,
        independent_saved_model_replay_exact=True, audit_used=False, submission_changed=False,
        cloud_prepared_manifest_sha256=file_hash(cloud_prepared_manifest),
        cloud_quantized_manifest_sha256=file_hash(cloud_quantized_manifest),
        cloud_completion_receipt_sha256=file_hash(receipts / 'complete.json'),
        caveat='Conditional internal development comparison, not a leaderboard estimate. Large cloud matrices are ephemeral; verified feature sources remain local.')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'report.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cloud-fit', 'cloud-prepared-manifest', 'cloud-quantized-manifest', 'receipts',
                 'prefix-expectation', 'source', 'full-tune', 'full-dev', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline-runs', nargs='+', required=True)
    run(**vars(parser.parse_args()))
