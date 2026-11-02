"""Predeclared small selective-fusion screen, selected exclusively on tune.

Calibration-only routing is a mandatory comparator: otherwise an improvement
from changing thresholds could be wrongly credited to multilingual features.
All outcomes remain exploratory and do not change production releases.
"""
import json
import numpy as np
from v3_transfer_data import PILOT, load, metric, owner_scores, bootstrap, write


def routes(a):
    p = a['baseline_200000']
    return {'none': np.zeros(len(p), dtype=bool), 'missing_address': a['missing_address'],
            'cross_script': a['cross_script'], 'uncertain_tree': (p >= .1) & (p < .95)}


def apply(a, new, policy):
    base = a['baseline_200000']
    selected = base >= .70
    mask = routes(a)[policy['route']]
    weight = policy['weight']
    selected[mask] = ((1-weight)*base[mask] + weight*new[mask]) >= policy['threshold']
    return selected.astype(np.float32)


def choose(a, new=None):
    base = a['baseline_200000']
    if new is None:
        new = base
        weights = [0.]
    else:
        weights = [0., .1, .25, .5]
    best = dict(route='none', weight=0., threshold=.70)
    best_key = (float(owner_scores(a, base, .70).mean()), 0, 0., .70)
    for route in ('missing_address', 'cross_script', 'uncertain_tree'):
        for weight in weights:
            for threshold in (.50, .60, .65, .70, .75, .80, .85, .90, .95):
                policy = dict(route=route, weight=weight, threshold=threshold)
                p = apply(a, new, policy)
                key = (float(owner_scores(a, p, .5).mean()), -1, -weight, threshold)
                if key > best_key:
                    best_key, best = key, policy
    best['tune_macro_f05'] = best_key[0]
    return best


def run():
    tuning, dev = load('tune'), load('dev')
    base_scores = owner_scores(dev, dev['baseline_200000'], .70)
    calibration = choose(tuning)
    calibration_p = apply(dev, dev['baseline_200000'], calibration)
    calibration_scores = owner_scores(dev, calibration_p, .5)
    report = dict(complete=True, exploratory=True,
        limitation='Adaptive diagnostic added after observing sampled-control gap and complementarity. Multiple route/weight/threshold searches on reused tuning; unused-owner confirmation required.',
        calibration_only=dict(policy=calibration, dev=metric(dev, calibration_p, .5), paired=bootstrap(base_scores, calibration_scores)),
        methods={})
    for tag in ('control', 'importance_control', 'e5', 'importance_e5', 'contrastive', 'indic_ascii_control', 'indic', 'xlmr'):
        if not (PILOT / tag / 'report.json').exists():
            continue
        policy = choose(tuning, np.load(PILOT / tag / 'tune_probabilities.npy'))
        p = apply(dev, np.load(PILOT / tag / 'dev_probabilities.npy'), policy)
        scores = owner_scores(dev, p, .5)
        report['methods'][tag] = dict(policy=policy, dev=metric(dev, p, .5),
            versus_200k=bootstrap(base_scores, scores), versus_calibration_only=bootstrap(calibration_scores, scores))
    write(PILOT / 'selective_fusion.json', report)


if __name__ == '__main__':
    run()
