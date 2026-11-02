"""Add a seventh learned channel to existing feature caches; preserve all inputs."""

import argparse
import hashlib
import json
import resource
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ber import expansion as expansion_module
from ber.data import read_tsv, write_json, write_tsv
from ber.disk_index import TargetStore
from ber.expansion import CHANNEL, ExpansionRetriever
from ber.features import pair_features
from ber.shards import ablate_arrays, file_hash, finalize_features, shard_paths
from ber.store import ensure_disk
from ber.text import Record


def worker(source, lexicon, output, expansion_k, workers, worker_id, minimum_free_gib):
    source, output = Path(source), Path(output)
    config = json.loads((source / "config.json").read_text())
    retriever = ExpansionRetriever(config["index"], lexicon, config["country_block"],
        config["gallery_fold"], config["max_df_ratio"], config["frequency_mode"])
    targets = TargetStore(config["store"])
    try:
        for shard_no, path in enumerate(shard_paths(source)):
            if shard_no % workers != worker_id:
                continue
            destination = output / "shards" / path.parent.name
            if (destination / "done.json").exists():
                continue
            ensure_disk(output, minimum_free_gib)
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            a["channel_order"] = np.asarray([*config["channel_order"], CHANNEL])
            for name in ("channel_scores", "channel_ranks"):
                a[name] = np.column_stack((a[name], np.zeros(len(a["y"]), dtype=a[name].dtype)))
            references = [Record.from_row(r) for r in read_tsv(path.parent / "references.tsv")]
            gold = ({r["source1_entity_id"]: set(filter(None, r["matched_entity_ids"].split(",")))
                    for r in read_tsv(path.parent / "ground_truth.tsv")} if config["labels_available"] else {})
            existing = {(int(q), str(t)): i for i, (q, t) in enumerate(zip(a["query_ids"], a["target_ids"]))}
            extra = {k: [] for k in ("X", "y", "query_ids", "target_ids", "channel_scores", "channel_ranks")}
            query_seconds = lookup_seconds = feature_seconds = 0.
            for q, reference in enumerate(references):
                stamp = time.perf_counter()
                hits = retriever.candidates(reference, expansion_k)
                query_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                records = targets.get_many([row_id for row_id, _ in hits])
                lookup_seconds += time.perf_counter() - stamp
                stamp = time.perf_counter()
                for row_id, (score, reciprocal) in hits:
                    target = records[row_id]
                    key = (q, target.entity_id)
                    rank = round(1 / reciprocal)
                    if key in existing:
                        i = existing[key]
                        a["channel_scores"][i, -1], a["channel_ranks"][i, -1] = score, rank
                    else:
                        extra["X"].append(pair_features(reference, target, {}))
                        extra["y"].append(int(target.entity_id in gold[reference.entity_id]) if config["labels_available"] else -1)
                        extra["query_ids"].append(q)
                        extra["target_ids"].append(target.entity_id)
                        extra["channel_scores"].append([0.] * 6 + [score])
                        extra["channel_ranks"].append([0] * 6 + [rank])
                feature_seconds += time.perf_counter() - stamp
            for name, values in extra.items():
                if values:
                    dtype = str if name == "target_ids" else a[name].dtype
                    a[name] = np.concatenate((a[name], np.asarray(values, dtype=dtype)))
            a = ablate_arrays(a)  # Recompute the 8 retrieval features from all 7 channels.
            staging = Path(tempfile.mkdtemp(prefix=f".{path.parent.name}.", dir=output / "shards"))
            np.savez_compressed(staging / "features.npz", **a)
            for name in ("references.tsv", "ground_truth.tsv"):
                if (path.parent / name).exists():
                    shutil.copy2(path.parent / name, staging / name)
            groups = [[] for _ in references]
            for q, target_id in zip(a["query_ids"], a["target_ids"]):
                groups[q].append(str(target_id))
            write_tsv(staging / "candidate_pairs.tsv", (
                {"source1_entity_id": r.entity_id, "candidate_entity_ids": ",".join(ids)}
                for r, ids in zip(references, groups)), ["source1_entity_id", "candidate_entity_ids"])
            previous = json.loads((path.parent / "done.json").read_text())
            done = dict(shard_id=int(path.parent.name), reference_ids_sha256=previous["reference_ids_sha256"],
                references=len(references), candidate_pairs=len(a["y"]), added_pairs=len(extra["y"]),
                query_seconds=query_seconds, lookup_seconds=lookup_seconds, feature_seconds=feature_seconds,
                peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            write_json(staging / "done.json", done)
            staging.rename(destination)
            print(f"augment {output.name}: shard {path.parent.name}, +{len(extra['y'])} pairs, search {query_seconds:.1f}s", flush=True)
    finally:
        targets.close()


def run(source, lexicon, output, expansion_k=20, workers=4, resume=False, minimum_free_gib=10):
    source, lexicon, output = map(Path, (source, lexicon, output))
    if min(expansion_k, workers) < 1:
        raise ValueError("Positive expansion budget/workers required")
    if not json.loads((source / "manifest.json").read_text())["complete"]:
        raise ValueError("Base feature cache is incomplete")
    config = json.loads((source / "config.json").read_text())
    if CHANNEL in config["channels"] or len(config["channels"]) != 6:
        raise ValueError("Augmentation expects the original six-channel cache")
    if config["labels_available"] and config["gallery_fold"] != "train":
        fitted = {r["entity_id"] for r in read_tsv(lexicon / "training_reference_ids.tsv")}
        refs = {r["entity_id"] for p in shard_paths(source) for r in read_tsv(p.parent / "references.tsv")}
        if fitted & refs:
            raise ValueError("Lexicon training/evaluation reference leakage")
    provenance = dict(source_features=str(source.resolve()), source_config_sha256=file_hash(source / "config.json"),
        lexicon=str(lexicon.resolve()), lexicon_sha256=file_hash(lexicon / "lexicon.json"),
        expansion_code_sha256=file_hash(expansion_module.__file__), expansion_k=expansion_k)
    config.update(channels=[*config["channels"], CHANNEL], channel_order=[*config["channel_order"], CHANNEL],
                  top_k_per_channel=max(expansion_k, config["top_k_per_channel"]), expansion=provenance)
    signature = "|".join([config["retrieval_code_sha256"], provenance["lexicon_sha256"],
                          provenance["expansion_code_sha256"], str(expansion_k)])
    config["retrieval_code_sha256"] = hashlib.sha256(signature.encode()).hexdigest()
    if output.exists():
        if not resume or json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Output exists or configuration changed; use a fresh output or matching --resume")
    else:
        (output / "shards").mkdir(parents=True)
        write_json(output / "config.json", config)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, source, lexicon, output, expansion_k, workers, i, minimum_free_gib)
                   for i in range(workers)]
        for future in futures:
            future.result()
    report = finalize_features(output)
    write_json(output / f"execution.{time.time_ns()}.json", dict(workers=workers,
        wall_seconds=time.perf_counter() - started, note="Additional augmentation cost; original cache cost is separate."))
    print(json.dumps({"references": report["references"], "candidate_pairs": report["candidate_pairs"],
                      "wall_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "lexicon", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expansion-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=10)
    run(**vars(parser.parse_args()))
