"""Exact challenge metric, including singleton scoring."""

import numpy as np


def entity_f05(truth, predicted):
    truth, predicted = set(truth), set(predicted)
    if not truth:
        return float(not predicted)
    tp = len(truth & predicted)
    return 1.25 * tp / (0.25 * len(truth) + len(predicted))


def score_arrays(query_ids, labels, probabilities, true_counts, threshold):
    accepted = probabilities >= threshold
    n = len(true_counts)
    predicted = np.bincount(query_ids[accepted], minlength=n)
    tp = np.bincount(query_ids[accepted & (labels == 1)], minlength=n)
    denominator = .25 * true_counts + predicted
    scores = np.divide(1.25 * tp, denominator, out=np.ones(n, dtype=float),
                       where=denominator != 0)
    singleton = true_counts == 0
    return {
        "macro_f05": float(scores.mean()),
        "pair_precision": float(tp.sum() / predicted.sum()) if predicted.sum() else 0.0,
        "pair_recall": float(tp.sum() / true_counts.sum()) if true_counts.sum() else 0.0,
        "predicted_pairs": int(predicted.sum()),
        "true_positive_pairs": int(tp.sum()),
        "singleton_false_merge_rate": float((predicted[singleton] > 0).mean())
        if singleton.any() else None,
        "entity_scores": scores,
    }


def tune_threshold(query_ids, labels, probabilities, true_counts):
    # Include reject-all explicitly. F0.5 is optimized on tune entities only.
    values = np.unique(np.concatenate((np.linspace(0, 1, 101),
                                       np.quantile(probabilities, np.linspace(0, 1, 101))
                                       if len(probabilities) else [],
                                       [np.nextafter(1.0, 2.0)])))
    best, best_score = 0.5, -1.0
    for threshold in values:
        score = score_arrays(query_ids, labels, probabilities, true_counts,
                             threshold)["macro_f05"]
        if score >= best_score:  # deterministic tie break toward a higher threshold
            best, best_score = float(threshold), score
    return best, best_score


def retrieval_metrics(query_ids, labels, true_counts):
    n = len(true_counts)
    counts = np.bincount(query_ids, minlength=n)
    recovered = np.bincount(query_ids[labels == 1], minlength=n)
    positive = true_counts > 0
    oracle = np.divide(1.25 * recovered, .25 * true_counts + recovered,
                       out=np.ones(n, dtype=float),
                       where=(.25 * true_counts + recovered) != 0)
    return {
        "candidate_pairs": int(len(query_ids)),
        "mean_candidates": float(counts.mean()),
        "p95_candidates": float(np.quantile(counts, .95)),
        "max_candidates": int(counts.max()),
        "pair_recall": float(recovered.sum() / true_counts.sum())
        if true_counts.sum() else 1.0,
        "macro_recall_non_singletons": float((recovered[positive] /
                                            true_counts[positive]).mean())
        if positive.any() else 1.0,
        "all_matches_covered_non_singletons": float(
            (recovered[positive] == true_counts[positive]).mean())
        if positive.any() else 1.0,
        "oracle_macro_f05": float(oracle.mean()),
    }
