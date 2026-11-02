"""Entity-disjoint selections and restartable, bounded-memory feature shards."""

import hashlib
import json
import random
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from . import features as feature_module
from . import disk_index as index_module
from .data import FIELDS, GOLD_FIELDS, read_tsv, reservoir_add, write_json, write_tsv
from .disk_index import DiskRetriever, TargetStore
from .features import FEATURE_NAMES, pair_features
from .metrics import retrieval_metrics
from .retrieval import BASE_CHANNELS, ASCII_CHANNELS
from .store import batches, connect, ensure_disk, lookup, record_row
from .text import Record

ALL_CHANNELS = (*BASE_CHANNELS, *ASCII_CHANNELS)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def gold_for(db, reference_ids):
    result = {eid: set() for eid in reference_ids}
    for group in batches(reference_ids, 800):
        rows = db.execute(f"SELECT reference_id,target_id FROM labels WHERE reference_id "
                          f"IN ({','.join('?' for _ in group)})", group)
        for row in rows:
            result[row["reference_id"]].add(row["target_id"])
    return result


def select_references(store, output, train_size=12000, tune_size=3000,
                      dev_size=3000, audit_size=3000, seed=2027, exclude_sample=None):
    if min(train_size, tune_size, dev_size, audit_size) <= 0:
        raise ValueError("All selection sizes must be positive")
    store, output = Path(store), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    db = connect(store)
    excluded = set()
    if exclude_sample:
        for path in Path(exclude_sample).glob("*/test_source1.tsv"):
            excluded.update(r["entity_id"] for r in read_tsv(path))
    capacity = {"train": train_size, "tune": tune_size, "eval": dev_size + audit_size}
    samples, seen = {f: [] for f in capacity}, {f: 0 for f in capacity}
    rng = random.Random(seed)
    for row in db.execute("SELECT * FROM anchors ORDER BY rowid"):
        if row["entity_id"] in excluded:
            continue
        fold = row["fold"]
        if fold not in samples:
            raise ValueError("Selections require a labeled training store")
        seen[fold] += 1
        reservoir_add(samples[fold], record_row(row), seen[fold], capacity[fold], rng)
    if any(len(samples[f]) != capacity[f] for f in capacity):
        raise ValueError("Insufficient references in one or more folds")
    rng.shuffle(samples["eval"])
    samples["dev"], samples["audit"] = samples["eval"][:dev_size], samples.pop("eval")[dev_size:]
    report = {"store": str(store.resolve()), "selection_seed": seed,
              "excluded_prior_reference_ids": len(excluded), "sets": {},
              "audit_policy": "Reserve audit until all hyperparameters and thresholds are frozen"}
    for fold, records in samples.items():
        folder = output / fold
        folder.mkdir()
        records.sort(key=lambda r: r["entity_id"])
        gold = gold_for(db, [r["entity_id"] for r in records])
        write_tsv(folder / "references.tsv", records, FIELDS)
        write_tsv(folder / "ground_truth.tsv",
                  ({"source1_entity_id": r["entity_id"], "matched_entity_ids":
                    ",".join(sorted(gold[r["entity_id"]]))} for r in records), GOLD_FIELDS)
        report["sets"][fold] = {
            "references": len(records), "positive_pairs": sum(map(len, gold.values())),
            "singletons": sum(not values for values in gold.values()),
            "gallery_fold": "eval" if fold in ("dev", "audit") else fold,
            "references_sha256": file_hash(folder / "references.tsv"),
        }
    db.close()
    write_json(output / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def featurize_shards(store, index, references, output, gallery_fold=None, top_k=20,
                     transliteration=False, country_block=True, query_batch_size=200,
                     resume=False, stop_after_shards=None, unlabeled=False,
                     minimum_free_gib=10, workers=1, worker_id=0, initialize_only=False,
                     max_df_ratio=None, frequency_mode="drop", lexicon=None, gate_terms=0):
    if top_k < 1 or query_batch_size < 1:
        raise ValueError("top_k and query_batch_size must be positive")
    if workers < 1 or not 0 <= worker_id < workers:
        raise ValueError("Invalid worker assignment")
    store, index, references, output = map(Path, (store, index, references, output))
    store_config = json.loads((store / "config.json").read_text())
    if not unlabeled and store_config["split"] != "train":
        raise ValueError("Test data has no labels; use --unlabeled")
    lexicon_path = Path(lexicon).resolve() if lexicon else None
    lexicon_json = (lexicon_path / "lexicon.json") if lexicon_path else None
    if lexicon_path is not None and not lexicon_json.exists():
        raise FileNotFoundError(lexicon_json)
    use_gated = lexicon_path is not None
    if use_gated:
        from . import hybrid_retrieval as retrieval_mod
        from . import direct_features as direct_mod
        from .direct_features import CHANNELS
        channels = list(CHANNELS)
        retrieval_hash = file_hash(retrieval_mod.__file__)
        direct_hash = file_hash(direct_mod.__file__)
        frequency_mode = "gate"
        max_df_ratio = 0.01 if max_df_ratio is None else max_df_ratio
    else:
        channels = list(BASE_CHANNELS + (ASCII_CHANNELS if transliteration else ()))
        retrieval_hash = file_hash(index_module.__file__)
        direct_hash = None
    config = {
        "shard_version": 1, "store": str(store.resolve()), "index": str(index.resolve()),
        "references": str(references.resolve()), "references_sha256": file_hash(references),
        "index_config_sha256": file_hash(index / "config.json"), "gallery_fold": gallery_fold,
        "top_k_per_channel": top_k, "channels": channels, "country_block": country_block,
        "query_batch_size": query_batch_size, "labels_available": not unlabeled,
        "feature_names": FEATURE_NAMES, "channel_order": list(channels),
        "feature_code_sha256": file_hash(feature_module.__file__), "external_data_used": False,
        "retrieval_code_sha256": retrieval_hash,
        "direct_feature_code_sha256": direct_hash,
        "reference_count": sum(1 for _ in read_tsv(references)),
        "max_df_ratio": max_df_ratio,
        "frequency_mode": frequency_mode,
        "gate_terms": int(gate_terms),
        "lexicon": str(lexicon_path) if lexicon_path else None,
        "lexicon_sha256": file_hash(lexicon_json) if lexicon_json and lexicon_json.exists() else None,
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Feature configuration changed; use a new output")
    else:
        output.mkdir(parents=True)
        (output / "shards").mkdir()
        write_json(output / "config.json", config)
    if initialize_only:
        return {"initialized": True, "config": config}
    if use_gated:
        from .hybrid_retrieval import retrievers
        from .direct_features import make_arrays
        base, extra = retrievers(index, lexicon_path, int(gate_terms), gallery_fold, store=store)
        retriever = None
    else:
        base = extra = None
        retriever = DiskRetriever(index, channels=channels, country_block=country_block,
                                  fold=gallery_fold, max_df_ratio=max_df_ratio, frequency_mode=frequency_mode)
    targets = TargetStore(store)
    started, generated = time.perf_counter(), 0
    try:
        for shard_id, group in enumerate(batches(read_tsv(references), query_batch_size)):
            if shard_id % workers != worker_id:
                continue
            directory = output / "shards" / f"{shard_id:06d}"
            ids = [r["entity_id"] for r in group]
            signature = hashlib.sha256("\n".join(ids).encode()).hexdigest()
            if directory.exists():
                done = json.loads((directory / "done.json").read_text())
                if done["reference_ids_sha256"] != signature:
                    raise ValueError("Completed shard references changed")
                continue
            ensure_disk(output, minimum_free_gib)
            anchor_rows = lookup(targets.db, "anchors", "entity_id", ids)
            if len(anchor_rows) != len(ids):
                raise ValueError("Duplicate or unknown reference IDs")
            if any(record_row(anchor_rows[r["entity_id"]]) != r for r in group):
                raise ValueError("Reference content does not match this store")
            if gallery_fold and any(anchor_rows[eid]["fold"] != gallery_fold for eid in ids):
                raise ValueError("Reference and target fold differ; positives would be excluded")
            gold = gold_for(targets.db, ids) if not unlabeled else {eid: set() for eid in ids}
            input_sha = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
            if use_gated:
                arrays, timing = make_arrays(group, base, extra, targets, None if unlabeled else gold, top_k)
                query_seconds = timing["query_seconds"]
                lookup_seconds = timing["lookup_seconds"]
                feature_seconds = timing["feature_seconds"]
                labels = arrays["y"]
                by_q = [[] for _ in ids]
                for q, tid in zip(arrays["query_ids"], arrays["target_ids"]):
                    by_q[int(q)].append(str(tid))
                candidate_rows = [{"source1_entity_id": eid, "candidate_entity_ids": ",".join(by_q[i])}
                                  for i, eid in enumerate(ids)]
            else:
                ref_records = [Record.from_row(r) for r in group]
                X, labels, query_ids, target_ids, channel_scores, channel_ranks = [], [], [], [], [], []
                candidate_rows = []
                query_seconds = lookup_seconds = feature_seconds = 0.
                for q, reference in enumerate(ref_records):
                    stamp = time.perf_counter()
                    candidates = retriever.candidates(reference, top_k)
                    query_seconds += time.perf_counter() - stamp
                    stamp = time.perf_counter()
                    records = targets.get_many([i for i, _ in candidates])
                    lookup_seconds += time.perf_counter() - stamp
                    stamp = time.perf_counter()
                    eid = reference.entity_id
                    current = []
                    for row_id, hits in candidates:
                        target = records[row_id]
                        X.append(pair_features(reference, target, hits))
                        labels.append(int(target.entity_id in gold[eid]) if not unlabeled else -1)
                        query_ids.append(q)
                        target_ids.append(target.entity_id)
                        current.append(target.entity_id)
                        channel_scores.append([hits.get(c, (0., 0.))[0] for c in ALL_CHANNELS])
                        channel_ranks.append([round(1 / hits[c][1]) if c in hits else 0 for c in ALL_CHANNELS])
                    candidate_rows.append({"source1_entity_id": eid,
                                           "candidate_entity_ids": ",".join(current)})
                    feature_seconds += time.perf_counter() - stamp
                arrays = {
                    "X": np.asarray(X, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
                    "y": np.asarray(labels, dtype=np.int8),
                    "query_ids": np.asarray(query_ids, dtype=np.int32),
                    "target_ids": np.asarray(target_ids, dtype=str),
                    "reference_ids": np.asarray(ids),
                    "countries": np.asarray([r.country for r in ref_records]),
                    "true_counts": np.asarray([len(gold[eid]) if not unlabeled else -1 for eid in ids], dtype=np.int32),
                    "channel_scores": np.asarray(channel_scores, dtype=np.float32).reshape(-1, len(ALL_CHANNELS)),
                    "channel_ranks": np.asarray(channel_ranks, dtype=np.uint16).reshape(-1, len(ALL_CHANNELS)),
                }
            # Directory rename is the commit point. An interrupted staging
            # directory is preserved but ignored, never mixed with complete data.
            staging = Path(tempfile.mkdtemp(prefix=f".{shard_id:06d}.", dir=output / "shards"))
            np.savez_compressed(staging / "features.npz", **arrays)
            write_tsv(staging / "candidate_pairs.tsv", candidate_rows,
                      ["source1_entity_id", "candidate_entity_ids"])
            write_tsv(staging / "references.tsv", group, FIELDS)
            if not unlabeled:
                write_tsv(staging / "ground_truth.tsv",
                          ({"source1_entity_id": eid, "matched_entity_ids": ",".join(sorted(gold[eid]))}
                           for eid in ids), GOLD_FIELDS)
            outputs = ["features.npz", "candidate_pairs.tsv", "references.tsv"]
            if not unlabeled:
                outputs.append("ground_truth.tsv")
            done = {"shard_id": shard_id, "reference_ids_sha256": signature,
                    "input_sha256": input_sha,
                    "references": len(ids), "candidate_pairs": len(labels),
                    "query_seconds": query_seconds, "lookup_seconds": lookup_seconds,
                    "feature_seconds": feature_seconds,
                    "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "output_sha256": {name: file_hash(staging / name) for name in outputs}}
            write_json(staging / "done.json", done)
            staging.rename(directory)
            generated += 1
            print(f"features {output.name}: shard {shard_id}, {len(ids)} queries, "
                  f"{len(labels):,} pairs; retrieval {query_seconds:.2f}s, "
                  f"lookup {lookup_seconds:.2f}s, features {feature_seconds:.2f}s; "
                  f"peak RSS {done['peak_rss_MiB']:.0f} MiB.", flush=True)
            if stop_after_shards and generated >= stop_after_shards:
                return {"complete": False, "generated_shards": generated}
        if workers == 1:
            return finalize_features(output)
        return {"worker_complete": True, "worker_id": worker_id, "generated_shards": generated}
    finally:
        targets.close()


def finalize_features(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    paths = sorted(p for p in (output / "shards").glob("*/done.json")
                   if not p.parent.name.startswith("."))
    expected = (config["reference_count"] + config["query_batch_size"] - 1) // config["query_batch_size"]
    if [p.parent.name for p in paths] != [f"{i:06d}" for i in range(expected)]:
        raise ValueError("Missing or unexpected feature shard; resume workers before finalizing")
    summaries = [json.loads(p.read_text()) for p in paths]
    if sum(s["references"] for s in summaries) != config["reference_count"]:
        raise ValueError("Completed feature count differs from the input references")
    report = {
        "complete": True, "config": config, "shards": len(summaries),
        "references": sum(s["references"] for s in summaries),
        "candidate_pairs": sum(s["candidate_pairs"] for s in summaries),
        "query_seconds": sum(s["query_seconds"] for s in summaries),
        "lookup_seconds": sum(s["lookup_seconds"] for s in summaries),
        "feature_seconds": sum(s["feature_seconds"] for s in summaries),
        "peak_worker_rss_MiB": max((s["peak_rss_MiB"] for s in summaries), default=0),
        "gallery": json.loads((Path(config["store"]) / "manifest.json").read_text())["targets_by_fold_country"],
    }
    write_json(output / "manifest.json", report)
    return report


def shard_paths(directory):
    return sorted(p for p in (Path(directory) / "shards").glob("*/features.npz")
                  if not p.parent.name.startswith(".") and (p.parent / "done.json").exists())


def ablate_arrays(arrays, top_k=None, channels=None):
    """Exactly filter a cached top-k channel union and recompute retrieval features.

    Smaller-k channel results are prefixes of the cached rankings. No new index
    search is needed, so ties and corpus statistics remain fixed across ablations.
    """
    a = {k: arrays[k] for k in arrays.files} if hasattr(arrays, "files") else dict(arrays)
    ranks = a["channel_ranks"]
    channel_order = tuple(a.get("channel_order", ALL_CHANNELS))
    if ranks.shape[1] != len(channel_order):
        raise ValueError("Channel metadata and rank columns differ")
    active = ranks > 0
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        active &= ranks <= top_k
    if channels is not None:
        if not set(channels) <= set(channel_order):
            raise ValueError("Unknown ablation channel")
        active &= np.array([c in channels for c in channel_order])[None, :]
    keep = active.any(axis=1)
    for name in ("X", "y", "query_ids", "target_ids", "channel_scores", "channel_ranks"):
        a[name] = a[name][keep].copy()
    active = active[keep]
    scores = np.where(active, a["channel_scores"], 0.)
    ranks = np.where(active, a["channel_ranks"], 0)
    reciprocal = np.divide(1., ranks, out=np.zeros_like(scores), where=ranks > 0)
    start = FEATURE_NAMES.index("retrieval_channels")
    if len(a["X"]):
        a["X"][:, start:] = np.column_stack((
            active.sum(axis=1), scores.max(axis=1), reciprocal.max(axis=1),
            scores[:, 0], scores[:, 1], scores[:, 2], scores[:, 3], scores[:, 4:].max(axis=1)))
    a["channel_scores"], a["channel_ranks"] = scores, ranks
    return a


def load_shards(directory, top_k=None, channels=None, include_features=True, include_retrieval=False):
    config = json.loads((Path(directory) / "config.json").read_text())
    if top_k is not None and top_k > config["top_k_per_channel"]:
        raise ValueError("Ablation cannot exceed cached retrieval k")
    if channels is not None and not set(channels) <= set(config["channels"]):
        raise ValueError("Ablation requests a channel absent from cache")
    values, offset = {}, 0
    for path in shard_paths(directory):
        with np.load(path, allow_pickle=False) as raw:
            a = ablate_arrays(raw, top_k, channels)
        a["query_ids"] += offset
        offset += len(a["reference_ids"])
        names = ["X", "y", "query_ids", "target_ids", "true_counts", "reference_ids", "countries"]
        if include_retrieval:
            names.extend(("channel_scores", "channel_ranks"))
        for name in names:
            if name != "X" or include_features:
                values.setdefault(name, []).append(a[name])
    if not values:
        raise ValueError("No completed feature shards")
    result = {name: np.concatenate(parts) for name, parts in values.items()}
    if include_retrieval:
        result["channel_order"] = np.asarray(config["channel_order"])
    result["gold"] = {r["source1_entity_id"]: set(filter(None, r["matched_entity_ids"].split(",")))
                      for path in shard_paths(directory)
                      if (path.parent / "ground_truth.tsv").exists()
                      for r in read_tsv(path.parent / "ground_truth.tsv")}
    return result
