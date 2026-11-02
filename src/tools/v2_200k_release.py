"""Freeze an isolated 200k CPU release from verified full-tuning evidence only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ber.metrics import score_arrays, tune_threshold
from ber.shards import file_hash
from v2_cascade_cache import validate_inputs
from v2_fit_cpu import load
from v2_run_pipeline import NAMES, code_hashes


def read(path):
    return json.loads(Path(path).read_text())


def validate_tune_alignment(arrays, matrix):
    """Verify scoring labels and owner order against each immutable source shard."""
    for item in matrix['shards']:
        with np.load(Path(matrix['source']) / 'shards' / item['shard'] / 'features.npz',
                     allow_pickle=False) as shard:
            ps = slice(item['pair_start'], item['pair_end'])
            qs = slice(item['query_start'], item['query_end'])
            for key in ('y', 'query_ids'):
                expected = shard[key] + item['query_start'] if key == 'query_ids' else shard[key]
                if not np.array_equal(arrays[key][ps], expected):
                    raise ValueError('Tuning pair labels/order differ from source: ' + key)
            for key in ('reference_ids', 'true_counts', 'countries'):
                if not np.array_equal(arrays[key][qs], shard[key]):
                    raise ValueError('Tuning owner labels/order differ from source: ' + key)


def validate_metadata(config, report, training, quantized, training_hash, model_hash):
    if (config.get('training_references') != 200000
            or config.get('threshold_selection') != 'full_tune only'
            or config.get('audit_used') is not False or config.get('external_data_used') is not False):
        raise ValueError('Expected provided-data 200k fit selected on full tuning only')
    if (not training.get('complete') or training.get('references') != 200000
            or training.get('names') != NAMES or training['source_config']['gate_terms'] != 0
            or training['source_config']['gallery_fold'] != 'train'):
        raise ValueError('Wrong complete training matrix, gallery, or feature recipe')
    if (not quantized.get('complete') or quantized.get('references') != 200000
            or quantized.get('prepared_manifest_sha256') != training_hash
            or quantized.get('borders_learned_from') != 'training gallery only'
            or report.get('quantized_provenance') != quantized):
        raise ValueError('Quantization/training provenance mismatch')
    if report.get('features') != NAMES or report.get('model_sha256') != model_hash:
        raise ValueError('Model report or feature identity mismatch')
    threshold = report.get('threshold')
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid full-tuning threshold')
    if any(config.get(key) != value for key, value in
           dict(iterations=600, depth=8, learning_rate=.07, random_seed=2026).items()):
        raise ValueError('Unexpected controlled 200k training settings')
    return float(threshold)


def freeze(model_run, training_manifest, quantized_manifest, output,
           tune_matrix='artifacts/improvement_v2/cpu/prepared_gate0_full_tune_v1'):
    model_run, training_manifest, quantized_manifest, output, tune_matrix = map(
        Path, (model_run, training_manifest, quantized_manifest, output, tune_matrix))
    if output.exists():
        raise FileExistsError(output)
    config = read(model_run / 'config.json')
    report = read(model_run / 'enhanced_report.json')
    training, quantized = read(training_manifest), read(quantized_manifest)
    model_path = model_run / 'enhanced.cbm'
    model_hash = file_hash(model_path)
    threshold = validate_metadata(config, report, training, quantized, file_hash(training_manifest), model_hash)
    if training['feature_code_sha256'] != code_hashes()['v2_features']:
        raise ValueError('Current enhanced features differ from training')
    arrays, matrix = load(tune_matrix)
    if matrix['source_config']['gallery_fold'] is not None:
        raise ValueError('Full-tuning calibration requires the full training gallery')
    for key in ('gate_terms', 'index_config_sha256', 'lexicon_sha256', 'top_k_per_channel', 'channels'):
        if matrix['source_config'][key] != training['source_config'][key]:
            raise ValueError('Calibration/training recipe mismatch: ' + key)
    probabilities_path = model_run / 'enhanced_full_tune_probabilities.npy'
    _, _, binding, _, _, probabilities = validate_inputs(matrix['source'], probabilities_path,
        tune_matrix / 'manifest.json', store=matrix.get('resolved_store'), index=matrix.get('resolved_index'))
    if binding['model_sha256'] != model_hash:
        raise ValueError('Tuning probabilities came from another model')
    validate_tune_alignment(arrays, matrix)
    reproduced_threshold, _ = tune_threshold(arrays['query_ids'], arrays['y'], probabilities, arrays['true_counts'])
    metrics = score_arrays(arrays['query_ids'], arrays['y'], probabilities, arrays['true_counts'], threshold)
    if reproduced_threshold != threshold or abs(metrics['macro_f05'] - report['full_tune']['macro_f05']) > 1e-12:
        raise ValueError('Saved tuning predictions do not reproduce selected threshold/metric')
    model = CatBoostClassifier(thread_count=1)
    model.load_model(str(model_path))
    if list(model.feature_names_) != NAMES or model.tree_count_ != config['iterations']:
        raise ValueError('Loaded model schema/tree count differs')
    replay = model.predict_proba(arrays['X'], thread_count=1)[:, 1]
    if not np.array_equal(replay, probabilities):
        raise ValueError('Independent saved-model tuning replay differs')
    artifacts = {}

    def bind(name, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected regular release artifact: ' + str(path))
        path = path.resolve()
        artifacts[name] = dict(path=str(path), sha256=file_hash(path))

    for name, path in {
        'model': model_path, 'model_report': model_run / 'enhanced_report.json',
        'fit_config': model_run / 'config.json', 'prepared_train_manifest': training_manifest,
        'quantized_manifest': quantized_manifest, 'prepared_full_tune_manifest': tune_matrix / 'manifest.json',
        'full_tune_probabilities': probabilities_path,
        'full_tune_alignment': probabilities_path.with_suffix('.alignment.json'),
        'code_v2_cpu_features.py': Path(__file__).with_name('v2_cpu_features.py'),
        'code_v2_run_pipeline.py': Path(__file__).with_name('v2_run_pipeline.py'),
        'code_v2_200k_release.py': Path(__file__)}.items():
        bind(name, path)
    record = dict(schema_version=1, release_candidate='gate0_enhanced200k_cpu_v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), training_references=200000,
        model_sha256=model_hash, threshold=threshold, threshold_selection='full_tune only',
        owner_policy=dict(enabled=False, allowance=None, multiple_targets_per_reference_allowed=True,
                          unmatched_targets_allowed=True),
        expected_test_references=1732544, required_test_countries=['France', 'India', 'US'],
        artifacts=artifacts, final_audit_used_for_selection=False, external_data_used=False,
        evaluation_scope='Existing full-tuning/development evidence; no fresh 200k audit claimed',
        calibration_verification=dict(references=len(arrays['reference_ids']),
            threshold_replayed=True, model_predictions_exact=True,
            full_tune_macro_f05=metrics['macro_f05']),
        original_fit_config_paths=config['manifests'], original_metadata_modified=False,
        local_manifest_override=str(training_manifest.resolve()),
        status='Frozen 200k policy; full-test inference/assembly/validation still required',
        caveats=['No hidden-test or France accuracy claim',
                 'Existing 100k reserved audit is not consumed for this release',
                 'The 100k release and package are preserved separately'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(dict(release=str(output.resolve()), sha256=file_hash(output),
                         threshold=threshold, training_references=200000)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-run', 'training-manifest', 'quantized-manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--tune-matrix', default='artifacts/improvement_v2/cpu/prepared_gate0_full_tune_v1')
    freeze(**vars(parser.parse_args()))
