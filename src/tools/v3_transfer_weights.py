"""Exact inverse-inclusion weights for the predeclared negative sampler."""
import numpy as np
from v3_transfer_data import BASE, PILOT, load, write, digest


def inclusion_weights(selected, positives, hard, remaining_count, random_draws):
    if remaining_count and not random_draws:
        raise ValueError('Unobserved stratum cannot be importance corrected')
    certain = set(positives) | set(hard)
    return np.asarray([1. if int(i) in certain else remaining_count/random_draws for i in selected], dtype=np.float32)


def run():
    a = load('train')
    directory = BASE / 'prepared_gate0_train_20k_v1'
    original = {k: np.load(directory/f'{k}.npy', mmap_mode='r') for k in ('X', 'y', 'query_ids')}
    oq = original['query_ids']
    assert np.all(oq[:-1] <= oq[1:])
    assert np.all(a['query_ids'][:-1] <= a['query_ids'][1:])
    weights = np.ones(len(a['y']), dtype=np.float32)
    total = 0
    for qi in range(len(a['true_counts'])):
        lo, hi = np.searchsorted(a['query_ids'], [qi, qi+1])
        selected = a['original_pair_index'][lo:hi]
        global_q = int(oq[selected[0]])
        start, end = np.searchsorted(oq, [global_q, global_q+1])
        y = original['y'][start:end]
        pos = np.flatnonzero(y == 1) + start
        neg = np.flatnonzero(y == 0) + start
        strength = original['X'][neg, 9] + original['X'][neg, 12]
        hard = neg[np.argsort(-strength, kind='stable')[:4]]
        remaining = len(neg)-len(hard)
        draws = min(2, remaining)
        w = inclusion_weights(selected, pos, hard, remaining, draws)
        if not np.isclose(w.sum(), end-start):
            raise ValueError('Sampling design reconstruction mismatch')
        weights[lo:hi] = w
        total += end-start
    path = PILOT/'data/sampling_weights.npy'
    np.save(path, weights)
    write(PILOT/'data/sampling_weights_manifest.json', dict(complete=True, sample_pairs=len(weights),
        weighted_pairs=float(weights.sum()), full_candidate_pairs=int(total), minimum=float(weights.min()), maximum=float(weights.max()),
        data_manifest_sha256=digest(PILOT/'data/manifest.json'), weights_sha256=digest(path),
        scheme='All positives and 4 deterministic hard negatives have inclusion probability 1; other negatives sampled 2/(remaining negatives). Weight is inverse inclusion probability.',
        limitation='Corrects expected empirical loss, not finite-sample diversity or all CatBoost regularization/quantization effects.'))
    print(dict(sample_pairs=len(weights), full_pairs=total, weighted_pairs=float(weights.sum())), flush=True)


if __name__ == '__main__':
    run()
