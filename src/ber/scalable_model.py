"""Fit on bounded development data; predict arbitrary-size caches shard by shard."""

import gc
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from .data import read_tsv, write_json, write_tsv
from .experiment import evaluate, write_predictions
from .features import FEATURE_NAMES
from .context_features import CONTEXT_NAMES, model_matrix
from .metrics import retrieval_metrics, score_arrays, tune_threshold
from .shards import ALL_CHANNELS, ablate_arrays, file_hash, load_shards, shard_paths
from .store import ensure_disk


def _check_cache(directory, top_k, channels):
    directory = Path(directory)
    if not json.loads((directory / "manifest.json").read_text())["complete"]:
        raise ValueError("Feature cache must be complete")
    config = json.loads((directory / "config.json").read_text())
    if top_k is not None and top_k > config["top_k_per_channel"]:
        raise ValueError("Cannot exceed cached top_k")
    if channels is not None and not set(channels) <= set(config["channels"]):
        raise ValueError("Requested channel is not cached")
    return config


def _write_candidates(directory, arrays):
    groups = [[] for _ in arrays["reference_ids"]]
    for q, target in zip(arrays["query_ids"], arrays["target_ids"]):
        groups[q].append(str(target))
    write_tsv(directory / "candidate_pairs.tsv",
        ({"source1_entity_id": str(eid), "candidate_entity_ids": ",".join(ids)}
         for eid, ids in zip(arrays["reference_ids"], groups)),
        ["source1_entity_id", "candidate_entity_ids"])


def _combine_tables(output, name):
    # Only generated files in this run are combined; original files are untouched.
    paths = sorted((output / "shards").glob(f"[0-9]*/{name}"))
    if not paths:
        return
    destination = output / name
    # On resume, a previous interrupted assembly may exist; preserve it.
    if destination.exists():
        destination.rename(output / f"{name}.previous.{time.time_ns()}")
    with destination.open("x", encoding="utf-8", newline="") as handle:
        for i, path in enumerate(paths):
            with path.open(encoding="utf-8", newline="") as source:
                header = next(source)
                if i == 0:
                    handle.write(header)
                for line in source:
                    handle.write(line)


def predict_shards(features, model, threshold, output, top_k=None, channels=None,
                    threads=4, resume=False, stop_after_shards=None, minimum_free_gib=10):
    features, model, output = map(Path, (features, model, output))
    if not np.isfinite(threshold) or threads < 1:
        raise ValueError("Invalid threshold or thread count")
    cache = _check_cache(features, top_k, channels)
    config = {
        "features": str(features.resolve()), "cache_config_sha256": file_hash(features / "config.json"),
        "model": str(model.resolve()), "model_sha256": file_hash(model), "threshold": threshold,
        "top_k": top_k, "channels": list(channels) if channels is not None else cache["channels"],
        "labels_available": cache["labels_available"], "external_data_used": False,
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Prediction configuration changed")
    else:
        output.mkdir(parents=True)
        (output / "shards").mkdir()
        write_json(output / "config.json", config)
    classifier = CatBoostClassifier(thread_count=threads)
    classifier.load_model(str(model))
    started, generated = time.perf_counter(), 0
    for path in shard_paths(features):
        directory = output / "shards" / path.parent.name
        if directory.exists():
            if not (directory / "done.json").exists():
                raise ValueError("Prediction shard is not committed")
            continue
        ensure_disk(output, minimum_free_gib)
        with np.load(path, allow_pickle=False) as raw:
            a = ablate_arrays(raw, top_k, channels)
        matrix = model_matrix(a, classifier.feature_names_)
        probability = classifier.predict_proba(matrix, thread_count=threads)[:, 1] if len(a["y"]) else np.array([])
        staging = Path(tempfile.mkdtemp(prefix=f".{path.parent.name}.", dir=output / "shards"))
        np.save(staging / "probabilities.npy", probability)
        _write_candidates(staging, a)
        done = {"references": len(a["reference_ids"]), "candidate_pairs": len(a["y"])}
        if cache["labels_available"]:
            a["gold"] = {r["source1_entity_id"]: set(filter(None, r["matched_entity_ids"].split(",")))
                         for r in read_tsv(path.parent / "ground_truth.tsv")}
            write_predictions(staging, a, probability, threshold)
            raw_score = score_arrays(a["query_ids"], a["y"], probability, a["true_counts"], threshold)
            counts = np.bincount(a["query_ids"][probability >= threshold], minlength=len(a["reference_ids"]))
            tp = np.bincount(a["query_ids"][(probability >= threshold) & (a["y"] == 1)], minlength=len(counts))
            recovered = np.bincount(a["query_ids"][a["y"] == 1], minlength=len(counts))
            candidate_count = np.bincount(a["query_ids"], minlength=len(counts))
            write_tsv(staging / "entity_metrics.tsv", (
                dict(source1_entity_id=str(eid), country=str(country), true_pairs=int(true),
                     predicted_pairs=int(pred), true_positive_pairs=int(correct),
                     retrieved_true_pairs=int(hit), candidate_pairs=int(n), macro_f05=float(score))
                for eid, country, true, pred, correct, hit, n, score in zip(
                    a["reference_ids"], a["countries"], a["true_counts"], counts, tp, recovered,
                    candidate_count, raw_score["entity_scores"])),
                ["source1_entity_id", "country", "true_pairs", "predicted_pairs", "true_positive_pairs",
                 "retrieved_true_pairs", "candidate_pairs", "macro_f05"])
        else:
            accepted = [[] for _ in a["reference_ids"]]
            for q, target, p in zip(a["query_ids"], a["target_ids"], probability):
                if p >= threshold:
                    accepted[q].append(str(target))
            write_tsv(staging / "matching_results.tsv",
                ({"source1_entity_id": str(eid), "matched_entity_ids": ",".join(sorted(ids))}
                 for eid, ids in zip(a["reference_ids"], accepted)),
                ["source1_entity_id", "matched_entity_ids"])
        write_json(staging / "done.json", done)
        staging.rename(directory)
        generated += 1
        if stop_after_shards and generated >= stop_after_shards:
            return {"complete": False}
    for filename in ("candidate_pairs.tsv", "matching_results.tsv", "errors.tsv", "entity_metrics.tsv"):
        _combine_tables(output, filename)
    summaries = [json.loads(p.read_text()) for p in sorted((output / "shards").glob("[0-9]*/done.json"))]
    report = {"complete": True, "config": config,
              "references": sum(s["references"] for s in summaries),
              "candidate_pairs": sum(s["candidate_pairs"] for s in summaries),
              "last_invocation_seconds": round(time.perf_counter() - started, 3),
              "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if cache["labels_available"]:
        # Per-entity diagnostics are small for development runs. Prediction itself
        # remains sharded; no pair-feature matrix is loaded for the entire corpus.
        rows = list(read_tsv(output / "entity_metrics.tsv"))
        truth = np.array([int(r["true_pairs"]) for r in rows])
        pred = np.array([int(r["predicted_pairs"]) for r in rows])
        tp = np.array([int(r["true_positive_pairs"]) for r in rows])
        hit = np.array([int(r["retrieved_true_pairs"]) for r in rows])
        scores = np.array([float(r["macro_f05"]) for r in rows])
        counts = np.array([int(r["candidate_pairs"]) for r in rows])
        singleton = truth == 0
        report["matching"] = {
            "macro_f05": float(scores.mean()), "pair_precision": float(tp.sum() / max(1, pred.sum())),
            "pair_recall": float(tp.sum() / max(1, truth.sum())),
            "predicted_pairs": int(pred.sum()), "true_positive_pairs": int(tp.sum()),
            "singleton_false_merge_rate": float((pred[singleton] > 0).mean()) if singleton.any() else None,
            "by_country": {c: {"entities": sum(r["country"] == c for r in rows),
                "macro_f05": float(scores[[r["country"] == c for r in rows]].mean())}
                for c in sorted({r["country"] for r in rows})}}
        oracle = np.divide(1.25 * hit, .25 * truth + hit, out=np.ones(len(rows)), where=(.25 * truth + hit) != 0)
        report["retrieval"] = {
            "pair_recall": float(hit.sum() / max(1, truth.sum())),
            "oracle_macro_f05": float(oracle.mean()), "mean_candidates": float(counts.mean()),
            "p95_candidates": float(np.quantile(counts, .95)), "max_candidates": int(counts.max()),
            "all_matches_covered_non_singletons": float((hit[~singleton] == truth[~singleton]).mean())}
    write_json(output / "metrics.json", report)
    return report


def fit_shards(train_features, tune_features, output, dev_features=None, top_k=10,
                channels=None, iterations=400, depth=6, learning_rate=.07, threads=4, seed=2026,
                minimum_free_gib=10, context_features=False):
    output = Path(output)
    if min(iterations, depth, threads) < 1:
        raise ValueError("Invalid model parameters")
    caches = {name: _check_cache(path, top_k, channels) for name, path in
              (("train", train_features), ("tune", tune_features))}
    paths = {"train": train_features, "tune": tune_features}
    if dev_features is not None:
        caches["dev"] = _check_cache(dev_features, top_k, channels)
        paths["dev"] = dev_features
    if channels is None:
        channels = tuple(caches["train"]["channels"])
    for config_key in ("store", "index_config_sha256", "feature_code_sha256", "retrieval_code_sha256",
                       "max_df_ratio", "frequency_mode", "country_block"):
        if any(caches["train"][config_key] != config[config_key] for config in caches.values()):
            raise ValueError(f"Training/evaluation cache mismatch: {config_key}")
    if any(not set(channels) <= set(config["channels"]) for config in caches.values()):
        raise ValueError("Requested training channels absent from another cache")
    if any(not c["labels_available"] for c in caches.values()):
        raise ValueError("Fitting requires labels")
    if caches["train"]["gallery_fold"] != "train" or caches["tune"]["gallery_fold"] != "tune":
        raise ValueError("Fitting requires group-disjoint train/tune target galleries")
    if "dev" in caches and caches["dev"]["gallery_fold"] != "eval":
        raise ValueError("Use separate prediction/calibration for full-gallery stress tests")
    # Audit reference separation before fitting or creating model artifacts.
    split_ids = {}
    for name, path in paths.items():
        values = [r["entity_id"] for shard in shard_paths(path)
                  for r in read_tsv(shard.parent / "references.tsv")]
        split_ids[name] = set(values)
        if len(values) != len(split_ids[name]):
            raise ValueError("Duplicate references within a feature cache")
    seen = set()
    for values in split_ids.values():
        if seen & values:
            raise ValueError("Reference leakage between model-fitting/evaluation splits")
        seen.update(values)
    output.mkdir(parents=True, exist_ok=False)
    ensure_disk(output, minimum_free_gib)
    started = time.perf_counter()
    config = dict(train_features=str(Path(train_features).resolve()),
                  tune_features=str(Path(tune_features).resolve()),
                  dev_features=str(Path(dev_features).resolve()) if dev_features else None,
                  top_k=top_k, channels=list(channels) if channels is not None else caches["train"]["channels"],
                  iterations=iterations, depth=depth, learning_rate=learning_rate,
                  threads=threads, seed=seed, context_features=context_features, external_data_used=False)
    write_json(output / "config.json", config)
    train = load_shards(train_features, top_k, channels)
    train_ids = set(train["reference_ids"])
    write_tsv(output / "training_reference_ids.tsv",
              ({"entity_id": str(eid)} for eid in sorted(train_ids)), ["entity_id"])
    if set(train["y"]) != {0, 1}:
        raise ValueError("Training requires positive and negative retrieved pairs")
    model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
        loss_function="Logloss", random_seed=seed, thread_count=threads,
        allow_writing_files=False, verbose=False)
    stamp = time.perf_counter()
    names = FEATURE_NAMES + (CONTEXT_NAMES if context_features else [])
    model.fit(Pool(model_matrix(train, names), train["y"], feature_names=names))
    training_seconds = time.perf_counter() - stamp
    model.save_model(str(output / "model.cbm"))
    train_report = retrieval_metrics(train["query_ids"], train["y"], train["true_counts"])
    del train
    gc.collect()
    tune = load_shards(tune_features, top_k, channels)
    if train_ids & set(tune["reference_ids"]):
        raise ValueError("Reference leakage between train and tune")
    probability = model.predict_proba(model_matrix(tune, names))[:, 1] if len(tune["y"]) else np.array([])
    threshold, tuning_score = tune_threshold(tune["query_ids"], tune["y"], probability, tune["true_counts"])
    tune_ids = set(tune["reference_ids"])
    del tune
    gc.collect()
    report = {"config": config, "threshold": threshold, "threshold_selected_on": "tune",
              "tuning_macro_f05": tuning_score, "training_seconds": training_seconds,
              "train_retrieval": train_report,
              "feature_importances": sorted(zip(names, map(float, model.feature_importances_)),
                                            key=lambda x: -x[1])}
    write_json(output / "threshold.json", {"threshold": threshold, "selected_on": "tune"})
    for name, path in (("tune", tune_features), ("dev", dev_features)):
        if path is None:
            continue
        if name == "dev":
            config_dev = _check_cache(path, top_k, channels)
            refs = {r["entity_id"] for shard in shard_paths(path)
                    for r in read_tsv(shard.parent / "references.tsv")}
            if refs & (train_ids | tune_ids):
                raise ValueError("Reference leakage into development evaluation")
        result = predict_shards(path, output / "model.cbm", threshold, output / name,
                                 top_k=top_k, channels=channels, threads=threads,
                                 minimum_free_gib=minimum_free_gib)
        report[name] = {k: result[k] for k in ("matching", "retrieval", "references", "candidate_pairs")}
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(output / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def calibrate_shards(tuning_features, model, output, top_k=None, channels=None, threads=4):
    """Retune only the decision threshold for a frozen model on labeled tuning refs."""
    tuning_features, model, output = map(Path, (tuning_features, model, output))
    config = _check_cache(tuning_features, top_k, channels)
    if not config["labels_available"]:
        raise ValueError("Threshold calibration needs provided tuning labels")
    training_ids_path = model.parent / "training_reference_ids.tsv"
    if not training_ids_path.exists():
        raise ValueError("Need saved training IDs to audit tuning/reference separation")
    training_ids = {r["entity_id"] for r in read_tsv(training_ids_path)}
    a = load_shards(tuning_features, top_k, channels)
    if set(a["reference_ids"]) & training_ids:
        raise ValueError("Tuning reference overlaps model-fitting references")
    output.mkdir(parents=True, exist_ok=False)
    classifier = CatBoostClassifier(thread_count=threads)
    classifier.load_model(str(model))
    probability = classifier.predict_proba(model_matrix(a, classifier.feature_names_),
                                            thread_count=threads)[:, 1] if len(a["y"]) else np.array([])
    threshold, score = tune_threshold(a["query_ids"], a["y"], probability, a["true_counts"])
    # Stability check: select on four fifths of tuning references and score the
    # omitted fifth. Model fitting is unchanged; these are not final-test results.
    grid = np.linspace(0., 1., 101)
    score_table = np.array([score_arrays(a["query_ids"], a["y"], probability,
                                         a["true_counts"], value)["entity_scores"] for value in grid])
    rng = np.random.default_rng(2026)
    assignment = np.empty(len(a["reference_ids"]), dtype=int)
    assignment[rng.permutation(len(assignment))] = np.arange(len(assignment)) % 5
    stability = []
    for fold in range(5):
        hold = assignment == fold
        if not hold.any() or hold.all():
            continue
        averages = score_table[:, ~hold].mean(axis=1)
        best = np.flatnonzero(averages == averages.max())[-1]
        stability.append({"held_out_fold": fold, "threshold": float(grid[best]),
                          "held_out_macro_f05": float(score_table[best, hold].mean())})
    report = {"threshold": threshold, "selected_on": "provided tuning references only",
              "tuning_macro_f05": score, "tuning_references": len(a["reference_ids"]),
              "reference_ids_disjoint_from_model_training": True,
              "model": str(model.resolve()), "model_sha256": file_hash(model),
              "features": str(tuning_features.resolve()), "top_k": top_k,
              "channels": list(channels) if channels else config["channels"],
              "gallery_fold": config["gallery_fold"], "threshold_stability_5_folds": stability,
              "note": "All-gallery calibration includes previously seen distractor targets; held-out references and their positive targets were not fitted."}
    write_json(output / "threshold.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
