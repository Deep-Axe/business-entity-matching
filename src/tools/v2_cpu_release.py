"""Freeze a tuning-selected CPU/owner policy with hash-bound local evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ber.shards import file_hash
from v2_cascade_cache import validate_inputs


def freeze(model_run, comparison, owner_tune, owner_dev, reverse_index, output):
    model_run, comparison = Path(model_run), Path(comparison)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model_run / 'config.json').read_text())
    model_report = json.loads((model_run / 'enhanced_report.json').read_text())
    report = json.loads((comparison / 'report.json').read_text())
    model_hash = file_hash(model_run / 'enhanced.cbm')
    if model_report['model_sha256'] != model_hash or report['compatibility']['model_sha256'] != model_hash:
        raise ValueError('Comparison/model hash mismatch')
    if config['training_references'] != 100000 or config['audit_used'] or config['external_data_used']:
        raise ValueError('Expected the audit-free, external-data-free 100k release candidate')
    training_manifest = Path(config['manifests']['train']) / 'manifest.json'
    training = json.loads(training_manifest.read_text())
    if (not training['complete'] or training['references'] != 100000
            or training['source_config']['gate_terms'] != 0
            or training['source_config']['gallery_fold'] != 'train'):
        raise ValueError('Expected complete matched gate-0 training provenance')
    if model_report['quantized_provenance']['prepared_manifest_sha256'] != file_hash(training_manifest):
        raise ValueError('Model training manifest changed')
    reverse_index = Path(reverse_index)
    if report['compatibility']['reverse_index_config_sha256'] != file_hash(reverse_index / 'config.json'):
        raise ValueError('Reverse-index mismatch')
    artifacts = {}

    def bind(name, path):
        path = Path(path).resolve()
        artifacts[name] = dict(path=str(path), sha256=file_hash(path))

    for name, filename in [('model', 'enhanced.cbm'), ('model_report', 'enhanced_report.json'), ('fit_config', 'config.json')]:
        bind(name, model_run / filename)
    for name, directory in config['manifests'].items():
        bind(f'prepared_{name}_manifest', Path(directory) / 'manifest.json')
    bind('comparison_report', comparison / 'report.json')
    bind('evaluation_reverse_index_config', reverse_index / 'config.json')
    bind('evaluation_reverse_index_manifest', reverse_index / 'manifest.json')
    for split, directory in [('tune', owner_tune), ('dev', owner_dev)]:
        directory = Path(directory)
        if file_hash(directory / 'manifest.json') != report['source_manifests'][split]:
            raise ValueError('Owner evidence manifest mismatch')
        bind(f'owner_{split}_manifest', directory / 'manifest.json')
        bind(f'owner_{split}_evidence', directory / 'evidence.npz')
    for split in ['full_tune', 'dev']:
        matrix_path = Path(config['manifests'][split]) / 'manifest.json'
        matrix = json.loads(matrix_path.read_text())
        _, _, binding, _, _, _ = validate_inputs(matrix['source'],
            model_run / f'enhanced_{split}_probabilities.npy', matrix_path,
            store=matrix.get('resolved_store'), index=matrix.get('resolved_index'))
        if binding['model_sha256'] != model_hash:
            raise ValueError('Probability alignment belongs to another model')
        bind(f'{split}_probabilities', model_run / f'enhanced_{split}_probabilities.npy')
        bind(f'{split}_alignment', model_run / f'enhanced_{split}_probabilities.alignment.json')
    for tool in ['v2_cpu_features.py', 'v2_fit_cpu.py', 'v2_owner_competition.py', 'v2_cpu_release.py']:
        bind(f'code_{tool}', Path(__file__).with_name(tool))
    selected = report['selected_on_tune_only']
    expected = max(report['grid'], key=lambda item: item['tune_macro_f05'])
    if selected != expected:
        raise ValueError('Selected policy differs from tuning-only maximum')
    allowance = selected['allowance']
    result = dict(
        schema_version=1, release_candidate='gate0_enhanced100k_owner_v1',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        status='policy frozen; final audit and full-test validation still required',
        model_sha256=model_hash, training_references=100000,
        threshold=float(selected['threshold']), threshold_selection='full tuning only',
        owner_policy=dict(enabled=allowance is not None, allowance=allowance,
            eligibility='all pairs with original model probability >= threshold; no route gate',
            reject_when='best_other_strength > own_strength + allowance',
            text_strength_storage_dtype='float32', unknown_competitor_strength=-1.,
            same_reference_excluded=True, arbitrary_tie_breaking=False,
            channels=['name_words', 'ascii_words', 'combined'],
            reverse_top_k_per_channel=10, country_block=True, max_df_ratio=.01,
            frequency_mode='gate', label_free=True,
            deployment_gallery='complete production Source1 text gallery, not training anchors',
            multiple_targets_per_reference_allowed=True, unmatched_targets_allowed=True,
            exhaustive_best_owner_assignment=False),
        comparison=dict(selected=selected,
            threshold_only=report['threshold_only_selected_on_tune'],
            tune=report['results']['tune'], dev=report['results']['dev']),
        artifacts=artifacts, final_audit_used_for_selection=False, external_data_used=False,
        caveats=['No hidden-test or French accuracy claim',
                 'Paired intervals condition on selected policy and repeated development use',
                 'Production reverse-index/store hashes must be bound separately before inference'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(dict(path=str(output.resolve()), sha256=file_hash(output),
                         threshold=result['threshold'], allowance=allowance)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-run', 'comparison', 'owner-tune', 'owner-dev', 'reverse-index', 'output']:
        parser.add_argument('--' + name, required=True)
    freeze(**vars(parser.parse_args()))
