"""US-to-India and India-to-US transfer checks; neither validates France."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from ber.context_features import CONTEXT_NAMES, model_matrix
from ber.data import write_json, write_tsv
from ber.features import FEATURE_NAMES
from ber.metrics import retrieval_metrics, score_arrays, tune_threshold
from ber.scalable_model import _check_cache, predict_shards
from ber.shards import load_shards


def country_subset(a, country):
    keep_queries = a["countries"] == country
    keep_pairs = keep_queries[a["query_ids"]]
    mapping = np.cumsum(keep_queries) - 1
    result = {k: a[k][keep_pairs] for k in ("X", "y", "target_ids")}
    result.update({k: a[k][keep_queries] for k in ("reference_ids", "countries", "true_counts")})
    result["query_ids"] = mapping[a["query_ids"][keep_pairs]]
    return result


def run(train_features, tune_features, dev_features, template_model, output, threads=4):
    output, template_model = Path(output), Path(template_model)
    template = json.loads((template_model / "config.json").read_text())
    k, channels = template["top_k"], template["channels"]
    train_config = _check_cache(train_features, k, channels)
    tune_config = _check_cache(tune_features, k, channels)
    dev_config = _check_cache(dev_features, k, channels)
    if [c["gallery_fold"] for c in (train_config, tune_config, dev_config)] != ["train", "tune", "eval"]:
        raise ValueError("Country transfer requires strict disjoint galleries")
    for key in ("store", "index_config_sha256", "feature_code_sha256", "retrieval_code_sha256", "max_df_ratio", "frequency_mode", "country_block"):
        if len({json.dumps(c[key], sort_keys=True) for c in (train_config, tune_config, dev_config)}) != 1:
            raise ValueError(f"Inconsistent feature caches: {key}")
    output.mkdir(parents=True, exist_ok=False)
    train = load_shards(train_features, k, channels)
    tune = load_shards(tune_features, k, channels)
    dev = load_shards(dev_features, k, channels, include_features=False)
    ids = [set(a["reference_ids"]) for a in (train, tune, dev)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Reference split leakage")
    del dev
    names = FEATURE_NAMES + (CONTEXT_NAMES if template.get("context_features") else [])
    reports = []
    for country in sorted(np.unique(train["countries"])):
        directory = output / country
        directory.mkdir()
        tr, tu = country_subset(train, country), country_subset(tune, country)
        model = CatBoostClassifier(iterations=template["iterations"], depth=template["depth"],
            learning_rate=template["learning_rate"], random_seed=template["seed"],
            thread_count=threads, allow_writing_files=False, verbose=False, loss_function="Logloss")
        started = time.perf_counter()
        model.fit(Pool(model_matrix(tr, names), tr["y"], feature_names=names))
        model.save_model(str(directory / "model.cbm"))
        write_tsv(directory / "training_reference_ids.tsv",
                  ({"entity_id": str(eid)} for eid in tr["reference_ids"]), ["entity_id"])
        probabilities = model.predict_proba(model_matrix(tu, names), thread_count=threads)[:, 1]
        threshold, score = tune_threshold(tu["query_ids"], tu["y"], probabilities, tu["true_counts"])
        write_json(directory / "config.json", {**template, "train_country": str(country),
            "threshold_tune_country": str(country), "purpose": "Country transfer stress test"})
        write_json(directory / "threshold.json", {"threshold": threshold, "selected_on": f"tune/{country}"})
        result = predict_shards(dev_features, directory / "model.cbm", threshold,
            directory / "dev", top_k=k, channels=channels, threads=threads)
        report = dict(train_country=str(country), train_references=len(tr["reference_ids"]),
            tune_references=len(tu["reference_ids"]), threshold=threshold, tune_macro_f05=score,
            dev_by_country=result["matching"]["by_country"], elapsed_seconds=time.perf_counter() - started)
        write_json(directory / "metrics.json", report)
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
        del tr, tu, model
        gc.collect()
    write_json(output / "summary.json", dict(experiments=reports,
        warning="Country transfer stress tests only. Training set sizes differ; France still has no labeled validation."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-features", "tune-features", "dev-features", "template-model", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(**vars(parser.parse_args()))
