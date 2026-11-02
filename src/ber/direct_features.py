"""Bounded in-memory seven-channel features for direct prediction or new training."""

import time

import numpy as np

from .expansion import CHANNEL
from .features import FEATURE_NAMES, pair_features
from .shards import ALL_CHANNELS, ablate_arrays
from .text import Record


CHANNELS = (*ALL_CHANNELS, CHANNEL)


def make_arrays(rows, base, extra, targets, gold=None, top_k=20):
    values = {name: [] for name in ("X", "y", "query_ids", "target_ids",
                                   "channel_scores", "channel_ranks")}
    timing = dict(query_seconds=0., lookup_seconds=0., feature_seconds=0.)
    records = [Record.from_row(row) for row in rows]
    for q, reference in enumerate(records):
        stamp = time.perf_counter()
        candidates = dict(base.candidates(reference, top_k))
        for row_id, hit in extra.candidates(reference, top_k):
            candidates.setdefault(row_id, {})[CHANNEL] = hit
        timing["query_seconds"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        found = targets.get_many(list(candidates))
        timing["lookup_seconds"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        for row_id, hits in candidates.items():
            target = found[row_id]
            values["X"].append(pair_features(reference, target, hits))
            values["y"].append(int(target.entity_id in gold[reference.entity_id]) if gold is not None else -1)
            values["query_ids"].append(q)
            values["target_ids"].append(target.entity_id)
            values["channel_scores"].append([hits.get(c, (0., 0.))[0] for c in CHANNELS])
            values["channel_ranks"].append([round(1 / hits[c][1]) if c in hits else 0 for c in CHANNELS])
        timing["feature_seconds"] += time.perf_counter() - stamp
    arrays = {
        "X": np.asarray(values["X"], dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
        "y": np.asarray(values["y"], dtype=np.int8),
        "query_ids": np.asarray(values["query_ids"], dtype=np.int32),
        "target_ids": np.asarray(values["target_ids"], dtype=str),
        "channel_scores": np.asarray(values["channel_scores"], dtype=np.float32).reshape(-1, len(CHANNELS)),
        "channel_ranks": np.asarray(values["channel_ranks"], dtype=np.uint16).reshape(-1, len(CHANNELS)),
        "channel_order": np.asarray(CHANNELS),
        "reference_ids": np.asarray([r.entity_id for r in records]),
        "countries": np.asarray([r.country for r in records]),
        "true_counts": np.asarray([len(gold[r.entity_id]) if gold is not None else -1 for r in records], dtype=np.int32),
    }
    return ablate_arrays(arrays), timing
