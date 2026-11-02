"""Tuning-only threshold calibration of a frozen matcher on a new retrieval gate.

This diagnostic does NOT claim matched-recipe training. All candidate contexts
are recomputed in the prepared target matrices before this script is called.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ber.data import write_json
from ber.metrics import tune_threshold
from ber.shards import file_hash
from v2_cascade_cache import validate_inputs, write_probability_binding
from v2_fit_cpu import ALL_NAMES, load, paired_bootstrap, score


def run(model, training_prepared, full_tune, full_dev, baseline_prepared,
        baseline_probabilities, baseline_threshold, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tune, tune_meta = load(full_tune)
    dev, dev_meta = load(full_dev)
    baseline, baseline_meta = load(baseline_prepared)
    _, _, binding, _, _, _ = validate_inputs(baseline_meta['source'], baseline_probabilities,
        Path(baseline_prepared) / 'manifest.json', store=baseline_meta.get('resolved_store'),
        index=baseline_meta.get('resolved_index'))
    if binding['model_sha256'] != file_hash(model):
        raise ValueError('Baseline probabilities were not produced by this frozen model')
    training_meta = json.loads((Path(training_prepared) / 'manifest.json').read_text())
    for key in ('gate_terms', 'index_config_sha256', 'lexicon_sha256', 'top_k_per_channel'):
        if tune_meta['source_config'][key] != dev_meta['source_config'][key]:
            raise ValueError('Tuning/development candidate recipes differ')
    if set(tune['reference_ids']) & set(dev['reference_ids']):
        raise ValueError('Tuning/development owners overlap')
    if set(dev['reference_ids']) != set(baseline['reference_ids']):
        raise ValueError('Paired comparison cohorts differ')
    trained_ids = np.load(Path(training_prepared) / 'reference_ids.npy', allow_pickle=False)
    if set(trained_ids) & (set(tune['reference_ids']) | set(dev['reference_ids'])):
        raise ValueError('Training owner leakage')
    classifier = CatBoostClassifier(thread_count=1)
    classifier.load_model(str(model))
    if classifier.feature_names_ != ALL_NAMES:
        raise ValueError('Frozen model feature schema differs')
    tp = classifier.predict_proba(tune['X'], thread_count=1)[:, 1]
    threshold, _ = tune_threshold(tune['query_ids'], tune['y'], tp, tune['true_counts'])
    tune_score, _ = score(tune, tp, threshold)
    dp = classifier.predict_proba(dev['X'], thread_count=1)[:, 1]
    dev_score, dev_scores = score(dev, dp, threshold)
    for split, probability, matrix in (('full_tune', tp, full_tune), ('dev', dp, full_dev)):
        path = output / f'{split}_probabilities.npy'
        np.save(path, probability)
        write_probability_binding(path, Path(matrix) / 'manifest.json', model)
    bp = np.load(baseline_probabilities, allow_pickle=False)
    baseline_score, baseline_scores = score(baseline, bp, baseline_threshold)
    by_id = dict(zip(map(str, baseline['reference_ids']), baseline_scores))
    aligned = np.asarray([by_id[str(eid)] for eid in dev['reference_ids']])
    report = dict(training_gate=training_meta['source_config']['gate_terms'],
        evaluation_gate=tune_meta['source_config']['gate_terms'], matched_recipe_retraining=False,
        diagnostic_only=True, threshold=threshold, threshold_selected_on='full_tune only',
        full_tune=tune_score, dev=dev_score, baseline_dev=baseline_score,
        baseline_threshold=baseline_threshold, paired_delta=paired_bootstrap(aligned, dev_scores),
        model=str(Path(model).resolve()), model_sha256=file_hash(model),
        training_manifest_sha256=file_hash(Path(training_prepared) / 'manifest.json'),
        full_tune_manifest_sha256=file_hash(Path(full_tune) / 'manifest.json'),
        full_dev_manifest_sha256=file_hash(Path(full_dev) / 'manifest.json'),
        audit_used=False, external_data_used=False)
    write_json(output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'training-prepared', 'full-tune', 'full-dev', 'baseline-prepared',
                 'baseline-probabilities', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--baseline-threshold', required=True, type=float)
    run(**vars(parser.parse_args()))
