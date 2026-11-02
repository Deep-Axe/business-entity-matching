"""Label-based retrieval/scoring diagnostics, with entity-level bootstrap intervals."""

import argparse
import json
from pathlib import Path

import numpy as np

from ber.data import read_tsv, write_json, write_tsv
from ber.features import FEATURE_NAMES
from ber.metrics import retrieval_metrics, score_arrays
from ber.shards import ALL_CHANNELS, ablate_arrays, load_shards
from ber.store import connect, lookup, record_row
from ber.text import Record


def summarize(a, probability, threshold, mask):
    n = len(a["reference_ids"])
    accepted = probability >= threshold
    predicted = np.bincount(a["query_ids"][accepted], minlength=n)
    correct = np.bincount(a["query_ids"][accepted & (a["y"] == 1)], minlength=n)
    recovered = np.bincount(a["query_ids"][a["y"] == 1], minlength=n)
    scores = score_arrays(a["query_ids"], a["y"], probability, a["true_counts"], threshold)["entity_scores"]
    truth = a["true_counts"][mask]
    return dict(entities=int(mask.sum()), macro_f05=float(scores[mask].mean()),
                pair_precision=float(correct[mask].sum() / max(1, predicted[mask].sum())),
                pair_recall=float(correct[mask].sum() / max(1, truth.sum())),
                blocking_recall=float(recovered[mask].sum() / max(1, truth.sum())),
                errors=int((scores[mask] < 1 - 1e-10).sum()))


def analyze(features, predictions, output, bootstrap=2000, seed=2029):
    features, predictions, output = map(Path, (features, predictions, output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((predictions / "config.json").read_text())
    if not config["labels_available"] or Path(config["features"]).resolve() != features.resolve():
        raise ValueError("Need labeled predictions for this exact feature cache")
    cached = load_shards(features, include_retrieval=True)
    a = ablate_arrays(cached, config["top_k"], config["channels"])
    probability = np.concatenate([np.load(p) for p in sorted((predictions / "shards").glob("[0-9]*/probabilities.npy"))])
    if len(probability) != len(a["y"]):
        raise ValueError("Pair alignment mismatch")
    report = dict(features=str(features.resolve()), predictions=str(predictions.resolve()),
                  threshold=config["threshold"], external_data_used=False)
    n = len(a["reference_ids"])
    cache_config = json.loads((features / "config.json").read_text())
    db = connect(cache_config["store"])
    anchors = lookup(db, "anchors", "entity_id", list(a["reference_ids"]))
    records = [Record.from_row(record_row(anchors[eid])) for eid in a["reference_ids"]]
    missing = np.array([not r.address for r in records])
    nonascii = np.array([not r.original_name.isascii() for r in records])
    scores = score_arrays(a["query_ids"], a["y"], probability, a["true_counts"], config["threshold"])["entity_scores"]
    slices = {"overall": np.ones(n, dtype=bool), "singleton": a["true_counts"] == 0,
              "non_singleton": a["true_counts"] > 0, "reference_address_missing": missing,
              "reference_address_present": ~missing, "reference_name_nonascii": nonascii,
              "reference_name_ascii": ~nonascii}
    slices.update({f"country_{country}": a["countries"] == country for country in np.unique(a["countries"])})
    report["slices"] = {name: summarize(a, probability, config["threshold"], mask)
                        for name, mask in slices.items() if mask.any()}
    rng = np.random.default_rng(seed)
    means = [float(scores[rng.integers(0, n, n)].mean()) for _ in range(bootstrap)]
    report["macro_f05_bootstrap_95_percent_interval"] = list(map(float, np.quantile(means, [.025, .975])))
    report["bootstrap_note"] = "Entity-resampling interval conditional on this model/gallery; not leaderboard or country-shift uncertainty."
    report["channel_coverage"] = {}
    for channel in cache_config["channels"]:
        subset = ablate_arrays(cached, cache_config["top_k_per_channel"], [channel])
        report["channel_coverage"][channel] = retrieval_metrics(subset["query_ids"], subset["y"], subset["true_counts"])
    report["k_coverage_selected_channels"] = {}
    for k in (5, 10, 20):
        if k <= cache_config["top_k_per_channel"]:
            subset = ablate_arrays(cached, k, config["channels"])
            report["k_coverage_selected_channels"][str(k)] = retrieval_metrics(subset["query_ids"], subset["y"], subset["true_counts"])
    bins = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (probability >= lo) & (probability < hi if hi < 1 else probability <= hi)
        bins.append(dict(lower=float(lo), upper=float(hi), pairs=int(selected.sum()),
                         mean_probability=float(probability[selected].mean()) if selected.any() else None,
                         positive_fraction=float(a["y"][selected].mean()) if selected.any() else None))
    report["calibration_bins"] = bins
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    false_positive = np.flatnonzero((probability >= config["threshold"]) & (a["y"] == 0))
    false_negative = np.flatnonzero((probability < config["threshold"]) & (a["y"] == 1))
    # A bounded, deterministic review of scored errors. Retrieval misses are
    # separately exported by predict-shards/errors.tsv, not hidden here.
    wrong = (sorted(false_positive, key=lambda i: (-float(probability[i]), str(a["target_ids"][i])))[:75]
             + sorted(false_negative, key=lambda i: (float(probability[i]), str(a["target_ids"][i])))[:75])
    report["scored_error_counts"] = {"false_positive_pairs": len(false_positive),
                                     "classifier_false_negative_pairs": len(false_negative)}
    targets = lookup(db, "records", "entity_id", [str(a["target_ids"][i]) for i in wrong])
    output.mkdir(parents=True)
    columns = ["source1_entity_id", "target_id", "country", "reference_name", "reference_address",
               "target_name", "target_address", "label", "probability", "decision", "name_ratio", "address_ratio", "number_jaccard"]
    rows = []
    for i in wrong:
        reference = anchors[a["reference_ids"][a["query_ids"][i]]]
        target = targets[a["target_ids"][i]]
        rows.append(dict(source1_entity_id=reference["entity_id"], target_id=target["entity_id"],
            country=reference["country"], reference_name=reference["name"], reference_address=reference["address"],
            target_name=target["name"], target_address=target["address"], label=int(a["y"][i]),
            probability=float(probability[i]), decision=int(probability[i] >= config["threshold"]),
            **{name: float(a["X"][i, names[name]]) for name in ("name_ratio", "address_ratio", "number_jaccard")}))
    write_tsv(output / "scored_errors.tsv", rows, columns)
    db.close()
    write_json(output / "report.json", report)
    print(json.dumps({"slices": report["slices"], "macro_f05_bootstrap_95_percent_interval": report["macro_f05_bootstrap_95_percent_interval"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "predictions", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2029)
    analyze(**vars(parser.parse_args()))
