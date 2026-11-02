"""Measure hypothetical expensive-scorer routing without training an oracle."""

import argparse
import json
from pathlib import Path

import numpy as np

from ber.data import write_json
from ber.features import FEATURE_NAMES
from ber.metrics import score_arrays
from ber.shards import ablate_arrays, shard_paths


def analyze(features, predictions, output):
    features, predictions, output = map(Path, (features, predictions, output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((predictions / "config.json").read_text())
    if not config["labels_available"]:
        raise ValueError("Routing diagnostics require provided labels")
    policies = {
        "probability_0.1_to_0.9": (.1, .9, False),
        "probability_0.01_to_0.99": (.01, .99, False),
        "probability_0.1_to_0.9_or_missing_address": (.1, .9, True),
    }
    totals = {name: dict(routed_pairs=0, retrieved_pairs=0, wrong_decisions=0,
                        wrong_decisions_routed=0, entity_scores_sum=0., references=0)
              for name in policies}
    for path in shard_paths(features):
        with np.load(path, allow_pickle=False) as raw:
            arrays = ablate_arrays(raw, config["top_k"], config["channels"])
        probabilities = np.load(predictions / "shards" / path.parent.name / "probabilities.npy")
        if len(probabilities) != len(arrays["y"]):
            raise ValueError("Prediction/cache alignment mismatch")
        wrong = (probabilities >= config["threshold"]) != (arrays["y"] == 1)
        for name, (lower, upper, include_missing) in policies.items():
            routed = (probabilities >= lower) & (probabilities <= upper)
            if include_missing:
                routed |= arrays["X"][:, FEATURE_NAMES.index("target_address_missing")] > .5
            # This is explicitly an upper bound: a perfect expensive scorer on
            # routed, already-retrieved pairs. It cannot recover retrieval misses.
            hypothetical = probabilities.copy()
            hypothetical[routed] = np.where(arrays["y"][routed] == 1,
                                             max(1., config["threshold"]), -1.)
            score = score_arrays(arrays["query_ids"], arrays["y"], hypothetical,
                                 arrays["true_counts"], config["threshold"])
            total = totals[name]
            total["routed_pairs"] += int(routed.sum())
            total["retrieved_pairs"] += len(routed)
            total["wrong_decisions"] += int(wrong.sum())
            total["wrong_decisions_routed"] += int((wrong & routed).sum())
            total["entity_scores_sum"] += float(score["entity_scores"].sum())
            total["references"] += len(arrays["reference_ids"])
    for total in totals.values():
        total["fraction_routed"] = total["routed_pairs"] / max(1, total["retrieved_pairs"])
        total["fraction_wrong_decisions_routed"] = total["wrong_decisions_routed"] / max(1, total["wrong_decisions"])
        total["perfect_routed_scorer_oracle_macro_f05"] = total.pop("entity_scores_sum") / total["references"]
    report = {"policies": totals,
              "warning": "Oracle upper bounds, not achieved model scores. No external lookup evidence used."}
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    analyze(**vars(parser.parse_args()))
