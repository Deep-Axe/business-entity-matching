"""Exclusive owner assignment: each source-2 record is claimed by at most one source-1."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v1_conflict_resolution import pair_features, prep_address, prep_name

TEXT_COLUMNS = [
    "name_set", "name_sort", "name_exact", "legal_agree", "addr_set", "addr_jaccard",
    "target_addr_missing", "house_in_target", "numbers_conflict", "numbers_missing",
]
FEATURE_CODE = Path(__file__).with_name("v1_conflict_resolution.py")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_features(left, right):
    return pair_features(
        prep_name(left["name"]), prep_address(left["address"]),
        prep_name(right["name"]), prep_address(right["address"]),
    )


def load_text(path):
    result = json.loads(Path(path).read_text())
    if not result.get("complete") or result["columns"] != TEXT_COLUMNS or result["training_fold"] != "train":
        raise ValueError("Unverified text scorer")
    if result["feature_code_sha256"] != digest(FEATURE_CODE):
        raise ValueError("Text scorer implementation changed")
    return result


def text_logit(left, right, model):
    return float(np.dot(text_features(left, right), model["coefficients"]) + model["intercept"])


def combined(p, text, alpha):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)) + alpha * np.asarray(text, dtype=float)


def winner(probabilities, text, alpha=0., floor=0., margin=0.):
    """Resolve a multi-claimant group. Exact ties abstain."""
    if len(probabilities) < 2:
        raise ValueError("Only conflicting claimants belong here")
    score = combined(probabilities, text, alpha)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite owner score")
    order = np.argsort(-score, kind="stable")
    first, second = order[:2]
    keep_floor = -math.inf if floor <= 0 else math.log(floor / (1 - floor))
    if score[first] < keep_floor or score[first] - score[second] <= max(0., margin):
        return None
    return int(first)
