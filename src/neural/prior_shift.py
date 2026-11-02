"""Saerens-style EM estimate of the positive rate among candidates, per country, on unlabeled test predictions."""
import glob, json, random, collections
import numpy as np

def em(p, prior_train, iters=200):
    pi = prior_train
    for _ in range(iters):
        a = p * pi / prior_train; b = (1 - p) * (1 - pi) / (1 - prior_train)
        q = a / (a + b); new = q.mean()
        if abs(new - pi) < 1e-7: break
        pi = new
    return pi

def implied_threshold(t, prior_train, pi):
    """Original-scale probability whose shifted posterior equals t."""
    k = (pi / prior_train) / ((1 - pi) / (1 - prior_train))
    return t / (t + k * (1 - t))

dev_p = np.load('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_dev_probabilities.npy')
dev_y = np.load('artifacts/improvement_v2/cpu/prepared_gate0_full_dev_v1/y.npy')
tune_p = np.load('artifacts/improvement_v2/cpu/fit_gate0_100k_v1/enhanced_full_tune_probabilities.npy')
prior = dev_y.mean()
print(f'dev: true positive rate among candidates {prior:.4f}, mean p {dev_p.mean():.4f}, EM on tune (sanity) {em(tune_p, prior):.4f}')

shards = sorted(glob.glob('artifacts/improvement_v2/inference/submission_controller_v1/collected/submission_100k_*/shards/*/'))
random.seed(0)
by = collections.defaultdict(list)
for sd in random.sample(shards, 600):
    z = np.load(sd + 'pair_ids.npz', allow_pickle=True); p = np.load(sd + 'probabilities.npy')
    c = z['countries'][z['query_ids']]
    for k in set(c):
        by[str(k)].append(p[c == k])
for c, v in sorted(by.items()):
    p = np.concatenate(v); pi = em(p, prior)
    print(f'{c:7s} pairs {len(p):8d} mean p {p.mean():.4f}  EM positive rate {pi:.4f} (x{pi / prior:.2f} train)  '
          f'threshold equivalent to 0.76 under shift: {implied_threshold(0.76, prior, pi):.3f}')
