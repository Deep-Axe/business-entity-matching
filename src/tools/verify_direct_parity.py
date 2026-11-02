"""Compare direct in-memory features with the existing frozen seven-channel cache."""

import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ber.context_features import model_matrix
from ber.data import read_tsv, write_json
from ber.deadline_retrieval import retrievers
from ber.direct_features import make_arrays
from ber.disk_index import TargetStore
from ber.shards import ablate_arrays


def run(cache, model, output):
    cache = Path(cache)
    config = json.loads((cache / "config.json").read_text())
    base, extra = retrievers(config["index"], config["expansion"]["lexicon"], 0)
    targets = TargetStore(config["store"])
    shard = cache / "shards/000000"
    rows = list(read_tsv(shard / "references.tsv"))
    direct, timing = make_arrays(rows, base, extra, targets)
    with np.load(shard / "features.npz", allow_pickle=False) as z:
        cached = ablate_arrays(z, top_k=20)
    classifier = CatBoostClassifier(thread_count=1)
    classifier.load_model(model)
    keys = lambda a: [(int(q), str(t)) for q, t in zip(a["query_ids"], a["target_ids"])]
    old_keys, new_keys = keys(cached), keys(direct)
    assert set(old_keys) == set(new_keys), "Candidate identity mismatch"
    positions = {key: i for i, key in enumerate(new_keys)}
    order = [positions[key] for key in old_keys]
    for field in ("X", "channel_scores", "channel_ranks"):
        np.testing.assert_allclose(cached[field], direct[field][order], rtol=0, atol=0)
    old_matrix, new_matrix = model_matrix(cached, classifier.feature_names_), model_matrix(direct, classifier.feature_names_)
    np.testing.assert_allclose(old_matrix, new_matrix[order], rtol=0, atol=0)
    np.testing.assert_allclose(classifier.predict_proba(old_matrix), classifier.predict_proba(new_matrix)[order], rtol=0, atol=0)
    targets.close()
    report = dict(passed=True, references=len(rows), pairs=len(old_keys),
                  exact_feature_and_probability_parity=True, timing=timing)
    write_json(output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "model", "output"):
        parser.add_argument(f"--{name}", required=True)
    run(**vars(parser.parse_args()))
