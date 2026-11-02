"""Read-only model diagnostics; writes a new report, never changes model choices."""

import argparse
import json
from pathlib import Path

import numpy as np

from ber.data import read_tsv
from ber.metrics import score_arrays


def analyze(sample_fold, runs, output, seed=2026):
    sample_fold, output = Path(sample_fold), Path(output)
    if output.exists():
        raise FileExistsError(output)
    targets = {r["entity_id"]: r for source in (2, 3)
               for r in read_tsv(sample_fold / f"test_source{source}.tsv")}
    gold = {r["source1_entity_id"]: set(filter(None, r["matched_entity_ids"].split(",")))
            for r in read_tsv(sample_fold / "ground_truth.tsv")}
    checks = {
        "S2": lambda r: r["entity_id"].startswith("S2-"),
        "S3": lambda r: r["entity_id"].startswith("S3-"),
        "India": lambda r: r["country"] == "India",
        "US": lambda r: r["country"] == "US",
        "missing_target_address": lambda r: not r["business_address"].strip(),
        "nonascii_target_name": lambda r: not r["business_name"].isascii(),
        "devanagari_target_name": lambda r: any("\u0900" <= c <= "\u097f"
                                                 for c in r["business_name"]),
    }
    report, scores, previous_ids = {"runs": {}, "scope": "sampled gallery only"}, {}, None
    rng = np.random.default_rng(seed)
    resample = None
    for run in map(Path, runs):
        metrics = json.loads((run / "metrics.json").read_text())
        a = np.load(run / "eval/features.npz", allow_pickle=False)
        if previous_ids is not None and not np.array_equal(previous_ids, a["reference_ids"]):
            raise ValueError("Runs must share the same reference order for paired comparison")
        previous_ids = a["reference_ids"]
        probability = np.load(run / "eval/probabilities.npy", allow_pickle=False)
        accepted = probability >= metrics["threshold"]
        entity_scores = score_arrays(a["query_ids"], a["y"], probability,
                                     a["true_counts"], metrics["threshold"])["entity_scores"]
        if resample is None:
            resample = rng.integers(0, len(entity_scores), (2000, len(entity_scores)))
        slices = {}
        for name, predicate in checks.items():
            positive_count = sum(predicate(targets[t]) for ids in gold.values() for t in ids)
            selected = np.array([predicate(targets[str(t)]) for t in a["target_ids"]])
            retrieved = int(((a["y"] == 1) & selected).sum())
            predicted = int((selected & accepted).sum())
            tp = int((selected & accepted & (a["y"] == 1)).sum())
            slices[name] = {
                "true_pairs": positive_count, "retrieved_true_pairs": retrieved,
                "predicted_pairs": predicted, "true_positive_pairs": tp,
                "blocking_recall": retrieved / positive_count if positive_count else None,
                "pair_precision": tp / predicted if predicted else None,
                "pair_recall": tp / positive_count if positive_count else None,
            }
        scores[run.name] = entity_scores
        interval = np.quantile(entity_scores[resample].mean(axis=1), [.025, .975])
        report["runs"][run.name] = {
            "macro_f05": float(entity_scores.mean()),
            "bootstrap_entity_95_percentile_interval": interval.tolist(),
            "pair_slices": slices,
        }
    if len(scores) == 2:
        names = list(scores)
        difference = scores[names[1]] - scores[names[0]]
        report["paired_comparison"] = {
            "difference": f"{names[1]} minus {names[0]}",
            "mean_macro_f05_difference": float(difference.mean()),
            "bootstrap_entity_95_percentile_interval": np.quantile(
                difference[resample].mean(axis=1), [.025, .975]).tolist(),
        }
    report["notes"] = [
        "Intervals resample reference entities, conditional on these fixed sampled galleries.",
        "They do not account for gallery selection, threshold tuning or geographic transfer.",
        "Slices overlap. Non-ASCII includes accented Latin, not just non-Latin scripts.",
        "No metrics here are used to change either frozen model or its threshold.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-fold", required=True)
    parser.add_argument("--runs", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    analyze(**vars(parser.parse_args()))
