"""Matched classifier-country holdout; explicitly NOT strict zero-shot ER.

Retrieval and the upstream supervised alias lexicon stay fixed and were built
using the full training fold. Only classifier training and threshold selection
exclude the target country. This diagnoses classifier source sensitivity.
"""
import json
import time

import numpy as np
from catboost import CatBoostClassifier
from v3_transfer_data import PILOT, SEED, load, tune, metric, owner_scores, bootstrap, write, digest


def subset(a, owners):
    ids = np.flatnonzero(owners) if np.asarray(owners).dtype == bool else np.asarray(owners)
    mapping = np.full(len(a['true_counts']), -1, dtype=np.int32)
    mapping[ids] = np.arange(len(ids))
    keep = mapping[a['query_ids']] >= 0
    result = {k: v[ids] if k in ('countries', 'reference_ids', 'true_counts') else v[keep]
              for k, v in a.items()}
    result['query_ids'] = mapping[a['query_ids'][keep]]
    return result, keep


def run():
    out = PILOT / 'classifier_country_holdout'
    out.mkdir(exist_ok=False)
    arrays = {s: load(s) for s in ('train', 'tune', 'dev')}
    extra = {s: np.load(PILOT / 'e5_features' / f'{s}_features.npy') for s in arrays}
    rng = np.random.default_rng(SEED)
    report = dict(complete=False, limitations='Classifier-country holdout only. Upstream alias lexicon/retrieval used both training countries. NOT strict zero-shot language/country transfer. No French labels.', results={})
    for country in ('US', 'India'):
        owners = np.sort(rng.choice(np.flatnonzero(arrays['train']['countries'] == country), 2500, replace=False))
        train, train_rows = subset(arrays['train'], owners)
        tuning, tune_rows = subset(arrays['tune'], arrays['tune']['countries'] == country)
        other = 'India' if country == 'US' else 'US'
        target, target_rows = subset(arrays['dev'], arrays['dev']['countries'] == other)
        source, source_rows = subset(arrays['dev'], arrays['dev']['countries'] == country)
        controls = {}
        for method in ('lexical', 'e5'):
            start = time.monotonic()
            matrix = {s: a['X'] for s, a in [('train', train), ('tune', tuning), ('target', target), ('source', source)]}
            if method == 'e5':
                for split, original, keep in [('train', 'train', train_rows), ('tune', 'tune', tune_rows), ('target', 'dev', target_rows), ('source', 'dev', source_rows)]:
                    matrix[split] = np.column_stack([matrix[split], extra[original][keep]])
            model = CatBoostClassifier(iterations=450, depth=7, learning_rate=.08, loss_function='Logloss',
                random_seed=SEED, thread_count=1, verbose=False, allow_writing_files=False)
            model.fit(matrix['train'], train['y'])
            score, threshold = tune(tuning, model.predict_proba(matrix['tune'], thread_count=1)[:, 1])
            p = model.predict_proba(matrix['target'], thread_count=1)[:, 1]
            source_p = model.predict_proba(matrix['source'], thread_count=1)[:, 1]
            current = owner_scores(target, p, threshold)
            result = dict(train_country=country, threshold_country=country, target_country=other,
                train_owners=2500, train_pairs=len(train['y']), threshold=threshold, tuning_macro_f05=score,
                target=metric(target, p, threshold), source=metric(source, source_p, threshold), seconds=time.monotonic()-start)
            if method == 'lexical':
                controls[other] = current
            else:
                result['paired_vs_lexical'] = bootstrap(controls[other], current)
            tag = f'{country}_to_{other}_{method}'
            report['results'][tag] = result
            np.save(out / f'{tag}_probabilities.npy', p)
            model.save_model(str(out / f'{tag}.cbm'))
            write(out / 'progress.json', report)
            print(json.dumps({k: result[k] for k in ['train_country', 'target_country', 'threshold', 'seconds']}), flush=True)
    report.update(complete=True, data_manifest_sha256=digest(PILOT / 'data/manifest.json'))
    write(out / 'report.json', report)


if __name__ == '__main__':
    run()
