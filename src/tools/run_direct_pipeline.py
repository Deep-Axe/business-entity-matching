"""Resumable bounded-memory features or direct predictions, partitionable across hosts.

Original data and frozen artifacts are never modified. Each shard is atomically
committed with input/output hashes. This driver never marks a partial partition
as a complete challenge submission.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import resource
import tempfile
import time
import importlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ber.context_features import model_matrix
from ber.data import FIELDS, read_tsv, write_json, write_tsv
from ber.hybrid_retrieval import retrievers
from ber.direct_features import CHANNELS, make_arrays
from ber.disk_index import TargetStore
from ber.features import FEATURE_NAMES
from ber.metrics import score_arrays
from ber.shards import file_hash, gold_for
from ber.store import batches, ensure_disk, lookup, record_row


_state = None


def initialize(config):
    global _state
    base, extra = retrievers(config["index"], config["lexicon"], config["gate_terms"], config["gallery_fold"], store=config["store"])
    targets = TargetStore(config["store"])
    model = None
    if config["mode"] == "predict":
        model = CatBoostClassifier(thread_count=1)
        model.load_model(config["model"])
    _state = config, base, extra, targets, model


def process(task):
    shard_id, rows = task
    config, base, extra, targets, model = _state
    output = Path(config["output"])
    destination = output / "shards" / f"{shard_id:06d}"
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        done = json.loads((destination / "done.json").read_text())
        if done["input_sha256"] != signature:
            raise ValueError("Resume input changed")
        for name, digest in done["output_sha256"].items():
            if file_hash(destination / name) != digest:
                raise ValueError("Committed output changed")
        return dict(done, resumed=True)
    ensure_disk(output, config["minimum_free_gib"])
    stamp = time.perf_counter()
    ids = [r["entity_id"] for r in rows]
    anchors = lookup(targets.db, "anchors", "entity_id", ids)
    if len(anchors) != len(ids) or any(record_row(anchors[r["entity_id"]]) != r for r in rows):
        raise ValueError("Unknown/duplicate/modified input reference")
    if config["gallery_fold"] and any(anchors[eid]["fold"] != config["gallery_fold"] for eid in ids):
        raise ValueError("Reference owner fold differs from requested gallery")
    gold = gold_for(targets.db, ids) if config["labels_available"] else None
    if config.get("from_features"):
        cached = Path(config["from_features"]) / "shards" / f"{shard_id:06d}"
        original = json.loads((cached / "done.json").read_text())
        if original["input_sha256"] != signature:
            raise ValueError("Cached feature reference mismatch")
        if file_hash(cached / "features.npz") != original["output_sha256"]["features.npz"]:
            raise ValueError("Cached feature data corrupted")
        with np.load(cached / "features.npz", allow_pickle=False) as z:
            a = {key: z[key] for key in z.files}
        timing = dict(query_seconds=0., lookup_seconds=0., feature_seconds=0., reused_feature_seconds=original["wall_seconds"])
    else:
        a, timing = make_arrays(rows, base, extra, targets, gold, config["top_k_per_channel"])
    staging = Path(tempfile.mkdtemp(prefix=f".{shard_id:06d}.", dir=output / "shards"))
    filenames = []
    if config["mode"] == "features":
        np.savez_compressed(staging / "features.npz", **a)
        write_tsv(staging / "references.tsv", rows, FIELDS)
        filenames.extend(["features.npz", "references.tsv"])
        if gold is not None:
            write_tsv(staging / "ground_truth.tsv", (
                {"source1_entity_id": eid, "matched_entity_ids": ",".join(sorted(gold[eid]))} for eid in ids),
                ["source1_entity_id", "matched_entity_ids"])
            filenames.append("ground_truth.tsv")
    else:
        probability = model.predict_proba(model_matrix(a, model.feature_names_), thread_count=1)[:, 1] if len(a["y"]) else np.array([])
        candidates, matches = [[] for _ in ids], [[] for _ in ids]
        for q, target_id, p in zip(a["query_ids"], a["target_ids"], probability):
            candidates[q].append(str(target_id))
            if p >= config["threshold"]:
                matches[q].append(str(target_id))
        for filename, column, groups in (("candidate_pairs.tsv", "candidate_entity_ids", candidates),
                                         ("matching_results.tsv", "matched_entity_ids", matches)):
            write_tsv(staging / filename, ({"source1_entity_id": eid, column: ",".join(sorted(values))}
                       for eid, values in zip(ids, groups)), ["source1_entity_id", column])
            filenames.append(filename)
        if gold is not None:
            np.savez_compressed(staging / "evaluation.npz", query_ids=a["query_ids"], y=a["y"],
                probabilities=probability, true_counts=a["true_counts"], reference_ids=a["reference_ids"], countries=a["countries"])
            filenames.append("evaluation.npz")
    done = dict(shard_id=shard_id, input_sha256=signature,
                reference_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                references=len(ids), candidate_pairs=len(a["y"]),
                wall_seconds=time.perf_counter()-stamp,
                peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                output_sha256={name: file_hash(staging / name) for name in filenames}, **timing)
    write_json(staging / "done.json", done)
    staging.rename(destination)
    return done


def run(index, store, references, lexicon, output, gate_terms=0, mode="predict", model=None,
        threshold=.67, workers=8, query_batch_size=200, gallery_fold=None, labeled=False,
        partition=0, partitions=1, resume=False, minimum_free_gib=8, top_k=20, stop_after_shards=None, partition_list=None,
        from_features=None, shard_ids_file=None, stop_file=None):
    if workers < 1 or query_batch_size < 1 or partitions < 1 or not 0 <= partition < partitions:
        raise ValueError("Invalid concurrency/partition")
    if gate_terms < -1 or top_k < 1 or not np.isfinite(threshold):
        raise ValueError("Invalid retrieval/threshold")
    assigned = sorted(set(partition_list if partition_list is not None else [partition]))
    if not assigned or any(i < 0 or i >= partitions for i in assigned):
        raise ValueError("Invalid partition list")
    selected_shards = None
    if shard_ids_file:
        entries = json.loads(Path(shard_ids_file).read_text())
        if not isinstance(entries, list) or any(type(i) is not int or i < 0 for i in entries) or len(set(entries)) != len(entries):
            raise ValueError("Expected unique nonnegative shard IDs")
        selected_shards = set(entries)
    if mode == "predict" and not model:
        raise ValueError("Prediction requires a model")
    if from_features and mode != "predict":
        raise ValueError("Feature reuse is prediction-only")
    store_config = json.loads((Path(store) / "config.json").read_text())
    if labeled and store_config["split"] != "train":
        raise ValueError("Test labels do not exist")
    if gallery_fold and not labeled:
        raise ValueError("Unlabeled prediction requires the complete gallery")
    output = Path(output).resolve()
    config = dict(index=str(Path(index).resolve()), store=str(Path(store).resolve()),
        references=str(Path(references).resolve()), lexicon=str(Path(lexicon).resolve()), output=str(output),
        mode=mode, from_features=str(Path(from_features).resolve()) if from_features else None,
        model=str(Path(model).resolve()) if model else None,
        model_sha256=file_hash(model) if model else None, threshold=threshold,
        gate_terms=gate_terms, gallery_fold=gallery_fold, labels_available=labeled,
        top_k_per_channel=top_k, channels=list(CHANNELS), channel_order=list(CHANNELS),
        query_batch_size=query_batch_size, partition=partition, partitions=partitions, assigned_partitions=assigned,
        selected_shards=sorted(selected_shards) if selected_shards is not None else None,
        minimum_free_gib=minimum_free_gib, reference_count=sum(1 for _ in read_tsv(references)),
        references_sha256=file_hash(references), lexicon_sha256=file_hash(Path(lexicon) / "lexicon.json"),
        index_config_sha256=file_hash(Path(index) / "config.json"), external_data_used=False,
        feature_names=FEATURE_NAMES, feature_code_sha256=file_hash(importlib.import_module("ber.features").__file__),
        retrieval_code_sha256=file_hash(importlib.import_module("ber.hybrid_retrieval").__file__),
        direct_feature_code_sha256=file_hash(importlib.import_module("ber.direct_features").__file__),
        wand_code_sha256=file_hash(importlib.import_module("ber.wand_retrieval").__file__) if gate_terms == -1 else None,
        runner_code_sha256=file_hash(__file__), max_df_ratio=.01, frequency_mode="gate", country_block=True)
    if from_features:
        previous = json.loads((Path(from_features) / "config.json").read_text())
        for key in ("gate_terms", "gallery_fold", "labels_available", "top_k_per_channel", "channels",
                    "query_batch_size", "reference_count", "references_sha256", "lexicon_sha256",
                    "index_config_sha256", "feature_code_sha256", "retrieval_code_sha256", "direct_feature_code_sha256"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Cached feature recipe mismatch: {key}")
    if output.exists():
        if not resume or json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Existing output requires identical config and --resume")
    else:
        (output / "shards").mkdir(parents=True)
        write_json(output / "config.json", config)
    def tasks():
        count = 0
        for shard_id, rows in enumerate(batches(read_tsv(references), query_batch_size)):
            if stop_file and Path(stop_file).exists():
                break
            if shard_id % partitions not in assigned or (selected_shards is not None and shard_id not in selected_shards):
                continue
            if stop_after_shards is not None and count >= stop_after_shards:
                break
            count += 1
            yield shard_id, rows
    started, completed, count, pairs, generated = time.perf_counter(), [], 0, 0, 0
    source = iter(tasks())
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(config,),
            mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {}
        for _ in range(workers * 2):
            item = next(source, None)
            if item is not None:
                pending[pool.submit(process, item)] = item[0]
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=30)
            if not ready:
                print(json.dumps(dict(status="working", completed_references=count,
                    wall_seconds=time.perf_counter()-started)), flush=True)
            for future in ready:
                pending.pop(future)
                result = future.result()
                completed.append(result["shard_id"])
                count += result["references"]
                pairs += result["candidate_pairs"]
                generated += 0 if result.get("resumed") else result["references"]
                progress = dict(completed_references=count, candidate_pairs=pairs, shard_id=result["shard_id"],
                    wall_seconds=time.perf_counter()-started,
                    generated_queries_per_second=generated/max(time.perf_counter()-started, .001),
                    worker_peak_rss_MiB=result["peak_rss_MiB"])
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                item = next(source, None)
                if item is not None:
                    pending[pool.submit(process, item)] = item[0]
    expected = [i for i in range((config["reference_count"]+query_batch_size-1)//query_batch_size)
                if i % partitions in assigned and (selected_shards is None or i in selected_shards)]
    if selected_shards is not None and not selected_shards <= set(expected):
        raise ValueError("Selected shard IDs outside this reference/partition assignment")
    complete = sorted(completed) == expected
    report = dict(complete=complete, partition_complete=complete, complete_submission=False,
        references=count, candidate_pairs=pairs, partition=partition, partitions=partitions,
        wall_seconds=time.perf_counter()-started, config=config)
    write_json(output / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("index", "store", "references", "lexicon", "output"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--mode", choices=("features", "predict"), default="predict")
    parser.add_argument("--model")
    parser.add_argument("--from-features")
    parser.add_argument("--shard-ids-file")
    parser.add_argument("--stop-file", help="Graceful drain: stop scheduling new shards if this marker appears")
    parser.add_argument("--threshold", type=float, default=.67)
    parser.add_argument("--gate-terms", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=200)
    parser.add_argument("--gallery-fold", choices=("train", "tune", "eval"))
    parser.add_argument("--labeled", action="store_true")
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--partition-list", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--stop-after-shards", type=int)
    run(**vars(parser.parse_args()))
