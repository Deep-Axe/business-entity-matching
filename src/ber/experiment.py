"""Local baseline with separate training, threshold selection, and evaluation."""

import csv
import gc
import json
import resource
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, Pool

from .data import FOLDS, read_tsv, write_json, write_tsv
from .features import FEATURE_NAMES, pair_features
from .metrics import retrieval_metrics, score_arrays, tune_threshold
from .retrieval import Retriever
from .text import Record


def featurize(folder, output, top_k, transliteration, country_block, threads):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    refs = [Record.from_row(r) for r in read_tsv(folder / "test_source1.tsv")]
    targets = [Record.from_row(r) for source in (2, 3)
               for r in read_tsv(folder / f"test_source{source}.tsv")]
    gold = {r["source1_entity_id"]: set(filter(None, r["matched_entity_ids"].split(",")))
            for r in read_tsv(folder / "ground_truth.tsv")}
    if set(gold) != {r.entity_id for r in refs}:
        raise ValueError("Ground truth must cover exactly the reference IDs.")
    target_ids = {r.entity_id for r in targets}
    if len(target_ids) != len(targets):
        raise ValueError("Duplicate target IDs.")
    if any(not matches <= target_ids for matches in gold.values()):
        raise ValueError("A ground-truth target is missing from the local gallery.")
    retriever = Retriever(targets, output / "index", transliteration,
                          country_block, threads)
    build_seconds = time.perf_counter() - started
    print(f"{folder.name}: indexed {len(targets):,} targets "
          f"in {build_seconds:.1f}s.", flush=True)
    true_counts = np.array([len(gold[r.entity_id]) for r in refs], dtype=np.int32)
    features, labels, query_ids, pair_target_ids = [], [], [], []
    channel_hits = Counter()
    ascii_only_hits = 0
    query_seconds = feature_seconds = 0.0
    with (output / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for query_id, reference in enumerate(refs):
            query_start = time.perf_counter()
            candidates = retriever.candidates(reference, top_k)
            query_seconds += time.perf_counter() - query_start
            feature_start = time.perf_counter()
            candidate_ids = []
            for row_id, channels in candidates:
                target = targets[row_id]
                label = int(target.entity_id in gold[reference.entity_id])
                features.append(pair_features(reference, target, channels))
                labels.append(label)
                query_ids.append(query_id)
                pair_target_ids.append(target.entity_id)
                candidate_ids.append(target.entity_id)
                if label:
                    channel_hits.update(channels.keys())
                    if all(channel.startswith("ascii") for channel in channels):
                        ascii_only_hits += 1
            writer.writerow([reference.entity_id, ",".join(candidate_ids)])
            feature_seconds += time.perf_counter() - feature_start
            if (query_id + 1) % 500 == 0:
                print(f"{folder.name}: {query_id + 1:,}/{len(refs):,} queries, "
                      f"{len(labels):,} candidates.", flush=True)
    arrays = {
        "X": np.asarray(features, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
        "y": np.asarray(labels, dtype=np.uint8),
        "query_ids": np.asarray(query_ids, dtype=np.int32),
        "target_ids": np.asarray(pair_target_ids),
        "true_counts": true_counts,
        "reference_ids": np.asarray([r.entity_id for r in refs]),
        "countries": np.asarray([r.country for r in refs]),
    }
    report = retrieval_metrics(arrays["query_ids"], arrays["y"], true_counts)
    report.update(
        channel_pair_recall={name: hits / max(1, int(true_counts.sum()))
                             for name, hits in channel_hits.items()},
        positive_pairs_unique_to_ascii_channels=ascii_only_hits,
        index_and_load_seconds=round(build_seconds, 3),
        query_seconds=round(query_seconds, 3),
        feature_seconds=round(feature_seconds, 3),
        references=len(refs), targets=len(targets),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    np.savez_compressed(output / "features.npz", **arrays)
    write_json(output / "retrieval_metrics.json", report)
    # Keep labels for error explanations only; no gold targets are injected into
    # retrieved candidates or into matcher training pairs.
    arrays["gold"] = gold
    del retriever, targets, refs, features, labels, query_ids, pair_target_ids
    gc.collect()
    return arrays, report


def evaluate(arrays, probabilities, threshold):
    report = score_arrays(arrays["query_ids"], arrays["y"], probabilities,
                          arrays["true_counts"], threshold)
    scores = report.pop("entity_scores")
    report["by_country"] = {}
    for country in sorted(set(arrays["countries"])):
        selected = arrays["countries"] == country
        report["by_country"][country] = {
            "entities": int(selected.sum()), "macro_f05": float(scores[selected].mean())}
    report["by_match_count"] = {}
    for label, mask in [
        ("singleton", arrays["true_counts"] == 0),
        ("one_match", arrays["true_counts"] == 1),
        ("multiple_matches", arrays["true_counts"] > 1),
    ]:
        report["by_match_count"][label] = {
            "entities": int(mask.sum()),
            "macro_f05": float(scores[mask].mean()) if mask.any() else None}
    return report


def write_predictions(folder, arrays, probabilities, threshold):
    accepted = {str(eid): [] for eid in arrays["reference_ids"]}
    scored = {}
    for query_id, target_id, probability in zip(arrays["query_ids"],
                                               arrays["target_ids"], probabilities):
        reference_id = str(arrays["reference_ids"][query_id])
        target_id = str(target_id)
        scored.setdefault(reference_id, {})[target_id] = float(probability)
        if probability >= threshold:
            accepted[reference_id].append(target_id)
    write_tsv(folder / "matching_results.tsv",
              ({"source1_entity_id": eid, "matched_entity_ids": ",".join(sorted(ids))}
               for eid, ids in accepted.items()),
              ["source1_entity_id", "matched_entity_ids"])
    errors = []
    for reference_id, matches in accepted.items():
        expected = arrays["gold"][reference_id]
        for target_id in sorted(set(matches) - expected):
            errors.append({"source1_entity_id": reference_id,
                           "target_id": target_id, "error": "false_positive",
                           "probability": scored[reference_id][target_id]})
        for target_id in sorted(expected - set(matches)):
            probability = scored.get(reference_id, {}).get(target_id)
            errors.append({"source1_entity_id": reference_id, "target_id": target_id,
                           "error": "retrieval_miss" if probability is None
                           else "classifier_false_negative",
                           "probability": "" if probability is None else probability})
    write_tsv(folder / "errors.tsv", errors,
              ["source1_entity_id", "target_id", "error", "probability"])


def run_experiment(sample_dir, output, top_k=10, transliteration=False,
                   country_block=True, threads=4, iterations=400, seed=2026):
    sample_dir, output = Path(sample_dir), Path(output)
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    if top_k <= 0 or iterations <= 0 or threads <= 0:
        raise ValueError("top_k, iterations and threads must be positive.")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = dict(sample_dir=str(sample_dir.resolve()), top_k_per_channel=top_k,
                  transliteration_retrieval=transliteration,
                  country_block=country_block, threads=threads,
                  iterations=iterations, seed=seed,
                  scope="local sampled galleries, not a competition submission",
                  external_data_used=False,
                  versions={p: version(p) for p in
                            ("numpy", "catboost", "rapidfuzz", "tantivy", "anyascii")})
    write_json(output / "config.json", config)
    arrays, reports = {}, {}
    for fold in FOLDS:
        arrays[fold], reports[fold] = featurize(
            sample_dir / fold, output / fold, top_k, transliteration, country_block, threads)
    if len(set(arrays["train"]["y"])) < 2:
        raise ValueError("Training retrieval must contain both positive and negative pairs.")
    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=.07,
        loss_function="Logloss", random_seed=seed, thread_count=threads,
        allow_writing_files=False, verbose=False)
    train_start = time.perf_counter()
    # A fixed iteration budget avoids using held-out evaluation labels.
    model.fit(Pool(arrays["train"]["X"], arrays["train"]["y"],
                   feature_names=FEATURE_NAMES))
    training_seconds = time.perf_counter() - train_start
    model.save_model(str(output / "model.cbm"))
    tune = arrays["tune"]
    tune_probability = model.predict_proba(tune["X"])[:, 1] if len(tune["y"]) else np.array([])
    threshold, _ = tune_threshold(tune["query_ids"], tune["y"], tune_probability,
                                  tune["true_counts"])
    report = {
        "config": config, "sample_manifest": manifest,
        "threshold": threshold, "threshold_selected_on": "tune",
        "training_seconds": round(training_seconds, 3),
        "feature_names": FEATURE_NAMES, "retrieval": reports, "matching": {},
        "limitations": [
            "All galleries are sampled: full-pool accuracy and runtime are unmeasured.",
            "Only US and India have labeled evaluation; France transfer is unmeasured.",
            "Fixed hyperparameters; tune fold selects only the acceptance threshold.",
            "No external lookup observations enter this pipeline.",
        ],
    }
    for fold in ("tune", "eval"):
        a = arrays[fold]
        probability = tune_probability if fold == "tune" else (
            model.predict_proba(a["X"])[:, 1] if len(a["y"]) else np.array([]))
        np.save(output / fold / "probabilities.npy", probability)
        report["matching"][fold] = evaluate(a, probability, threshold)
        report["matching"][fold]["at_0.5"] = evaluate(a, probability, .5)
        write_predictions(output / fold, a, probability, threshold)
    report["feature_importances"] = sorted(
        zip(FEATURE_NAMES, map(float, model.feature_importances_)),
        key=lambda item: -item[1])
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["peak_rss_MiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
    write_json(output / "metrics.json", report)
    print(json.dumps({k: report[k] for k in
                      ("threshold", "training_seconds", "matching",
                       "elapsed_seconds", "peak_rss_MiB")}, indent=2), flush=True)
    return report
