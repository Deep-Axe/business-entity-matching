"""Optional label-free candidate-neighborhood comparisons for crowded galleries."""

import numpy as np

from .features import FEATURE_NAMES

CONTEXT_NAMES = ["context_log_candidates", "context_log_near_names", "context_log_exact_names",
                 "context_log_strong_addresses", "context_name_minus_best", "context_address_minus_best",
                 "context_log_joint_strong_evidence", "context_log_missing_address_near_names"]


def context_matrix(arrays):
    base, q = arrays["X"], arrays["query_ids"]
    n = len(arrays["reference_ids"])
    column = lambda name: base[:, FEATURE_NAMES.index(name)]
    name_score = np.maximum(column("name_ratio"), column("ascii_name_ratio"))
    address_score = column("address_ratio")
    near_name = name_score >= .9
    missing_address = column("target_address_missing") > .5
    strong_address = ((column("address_overlap") >= .9) &
                      (column("number_jaccard") >= .5) & ~missing_address)
    count = lambda mask: np.bincount(q, weights=mask, minlength=n)
    best_name, best_address = np.zeros(n), np.zeros(n)
    np.maximum.at(best_name, q, name_score)
    np.maximum.at(best_address, q, address_score)
    extra = np.column_stack((
        np.log1p(np.bincount(q, minlength=n))[q],
        np.log1p(count(near_name))[q],
        np.log1p(count(column("name_exact") > .5))[q],
        np.log1p(count(strong_address))[q],
        name_score - best_name[q], address_score - best_address[q],
        np.log1p(count(near_name & strong_address))[q],
        np.log1p(count(near_name & missing_address))[q],
    )).astype(np.float32)
    return np.column_stack((base, extra))


def model_matrix(arrays, feature_names):
    if list(feature_names) == FEATURE_NAMES:
        return arrays["X"]
    if list(feature_names) == FEATURE_NAMES + CONTEXT_NAMES:
        return context_matrix(arrays)
    raise ValueError("Unknown model feature schema")
