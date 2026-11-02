"""Matched CPU controls, domain weighting and added transfer features."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from v3_transfer_data import PILOT, BASE, SEED, load, write, digest, tune, metric, owner_scores, bootstrap


def balanced_weights(a):
    countries = a['countries'][a['query_ids']]
    groups = np.array([f'{c}:{s}' for c, s in zip(countries, a['target_source'])])
    values, counts = np.unique(groups, return_counts=True)
    mapping = dict(zip(values, len(groups) / (len(values) * counts)))
    return np.array([mapping[g] for g in groups], dtype=np.float32)


def fit(tag, extras=None, weighting=False, full_negatives=False, importance=False):
    out = PILOT / tag
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    arrays = {s: load(s) for s in ('train', 'tune', 'dev')}
    if full_negatives:
        if extras:
            raise ValueError('Extra features cover sampled pairs only')
        directory = BASE / 'prepared_gate0_train_20k_v1'
        refs = np.load(directory / 'reference_ids.npy')
        selected = np.flatnonzero(np.isin(refs, arrays['train']['reference_ids']))
        if not np.array_equal(refs[selected], arrays['train']['reference_ids']):
            raise ValueError('Full-negative owner alignment failed')
        old_q = np.load(directory / 'query_ids.npy', mmap_mode='r')
        remap = np.full(len(refs), -1, dtype=np.int32)
        remap[selected] = np.arange(len(selected))
        keep = remap[old_q] >= 0
        arrays['train']['X'] = np.load(directory / 'X.npy', mmap_mode='r')[keep]
        arrays['train']['y'] = np.load(directory / 'y.npy', mmap_mode='r')[keep]
        arrays['train']['query_ids'] = remap[old_q[keep]]
        arrays['train']['target_source'] = np.where(arrays['train']['X'][:, 27] > .5, 2, 3)
    matrices = {s: a['X'] for s, a in arrays.items()}
    extra_binding = None
    if extras:
        extra_manifest = PILOT / extras / 'manifest.json'
        metadata = json.loads(extra_manifest.read_text())
        if not metadata['complete'] or metadata['data_manifest_sha256'] != digest(PILOT / 'data/manifest.json'):
            raise ValueError('Extra features belong to a different/incomplete dataset')
        extra_binding = dict(manifest_sha256=digest(extra_manifest), feature_sha256={})
        for s in arrays:
            path = PILOT / extras / f'{s}_features.npy'
            x = np.load(path)
            extra_binding['feature_sha256'][s] = digest(path)
            if x.ndim == 1:
                x = x[:, None]
            if len(x) != len(arrays[s]['y']) or not np.isfinite(x).all():
                raise ValueError('Invalid extra feature alignment')
            matrices[s] = np.column_stack([matrices[s], x]).astype(np.float32)
    model = CatBoostClassifier(iterations=450, depth=7, learning_rate=.08,
        loss_function='Logloss', random_seed=SEED, thread_count=1, verbose=False,
        allow_writing_files=False, used_ram_limit='1000mb')
    weights = balanced_weights(arrays['train']) if weighting else None
    if importance:
        if full_negatives or weighting:
            raise ValueError('Use importance correction only with sampled pooled training in this screen')
        path = PILOT / 'data/sampling_weights.npy'
        binding = json.loads((PILOT / 'data/sampling_weights_manifest.json').read_text())
        if (not binding['complete'] or binding['data_manifest_sha256'] != digest(PILOT / 'data/manifest.json')
                or binding['weights_sha256'] != digest(path)):
            raise ValueError('Invalid importance weights')
        weights = np.load(path)
        if len(weights) != len(arrays['train']['y']) or (weights <= 0).any():
            raise ValueError('Invalid weight dimensions')
        weights = weights / weights.mean()
    model.fit(matrices['train'], arrays['train']['y'], sample_weight=weights)
    model.save_model(str(out / 'model.cbm'))
    probabilities = {s: model.predict_proba(matrices[s], thread_count=1)[:, 1] for s in ('tune', 'dev')}
    for s, p in probabilities.items():
        np.save(out / f'{s}_probabilities.npy', p)
    tune_score, threshold = tune(arrays['tune'], probabilities['tune'])
    report = dict(complete=True, tag=tag, threshold=threshold, tune_macro_f05=tune_score,
        dev=metric(arrays['dev'], probabilities['dev'], threshold),
        training_owners=len(arrays['train']['true_counts']), training_pairs=len(arrays['train']['y']),
        feature_count=matrices['train'].shape[1], balanced=weighting,
        full_negatives=full_negatives,
        importance_correction=importance,
        extra_binding=extra_binding,
        seconds=time.monotonic()-start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        data_manifest_sha256=digest(PILOT / 'data/manifest.json'), model_sha256=digest(out / 'model.cbm'))
    compare(arrays, probabilities, report)
    if importance and (PILOT / 'importance_control/report.json').exists():
        old = json.loads((PILOT / 'importance_control/report.json').read_text())
        p = np.load(PILOT / 'importance_control/dev_probabilities.npy')
        report['comparisons']['importance_control'] = bootstrap(
            owner_scores(arrays['dev'], p, old['threshold']),
            owner_scores(arrays['dev'], probabilities['dev'], threshold))
    write(out / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('tag', 'threshold', 'tune_macro_f05', 'dev', 'seconds')}), flush=True)


def compare(arrays, probabilities, report):
    dev = arrays['dev']
    current = owner_scores(dev, probabilities['dev'], report['threshold'])
    report['comparisons'] = {}
    for size, threshold in ((100000, .76), (200000, .70)):
        p = dev[f'baseline_{size}']
        report['comparisons'][f'baseline_{size}'] = dict(baseline=metric(dev, p, threshold),
            paired=bootstrap(owner_scores(dev, p, threshold), current))
    control_path = PILOT / 'control/report.json'
    if control_path.exists():
        old = json.loads(control_path.read_text())
        p = np.load(PILOT / 'control/dev_probabilities.npy')
        report['comparisons']['matched_control'] = bootstrap(owner_scores(dev, p, old['threshold']), current)
    # A tuned additive combination tests whether pilot signal complements the
    # production models. No fitting or policy choice is made on development.
    report['fusion'] = {}
    for size in (100000, 200000):
        best = None
        for weight in (0., .1, .25, .5, .75, 1.):
            p = (1-weight)*arrays['tune'][f'baseline_{size}'] + weight*probabilities['tune']
            score, threshold = tune(arrays['tune'], p)
            choice = (score, -weight, threshold)
            if best is None or choice > best:
                best = choice
        score, negative_weight, threshold = best
        weight = -negative_weight
        p = (1-weight)*dev[f'baseline_{size}'] + weight*probabilities['dev']
        old_th = .76 if size == 100000 else .70
        report['fusion'][str(size)] = dict(weight=weight, threshold=threshold, tune_macro_f05=score,
            dev=metric(dev, p, threshold), paired=bootstrap(owner_scores(dev, dev[f'baseline_{size}'], old_th), owner_scores(dev, p, threshold)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('--extras')
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--full-negatives', action='store_true')
    parser.add_argument('--importance', action='store_true')
    args = parser.parse_args()
    fit(args.tag, args.extras, args.balanced, args.full_negatives, args.importance)
