"""Immutable-release owner veto for committed v2 inference; separate frozen audit.

Postprocessing never reads label members, preserves every reference and all
candidates, and writes a new artifact. Only the explicit audit subcommand reads
provided labels, after verifying a frozen release and completed postprocessing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np
import tantivy

from ber.data import read_tsv, write_json, write_tsv
from ber.metrics import score_arrays
from ber.shards import file_hash
from ber.store import batches, connect, ensure_disk, lookup, record_row
from ber.text import Record
from v2_owner_competition import Competition, veto
import v2_owner_competition
from v2_run_pipeline import signature


RECIPE_KEYS = ("mode", "model_sha256", "model_report_sha256", "training_manifest_sha256",
    "feature_names", "component_sha256", "weighting_policy", "weighting_index_config_sha256",
    "weighting_target_records", "training_weighting_index_config_sha256", "store_config_sha256",
    "store_split", "index_config_sha256", "lexicon_sha256", "gate_terms", "top_k_per_channel",
    "channels", "channel_order", "country_block", "max_df_ratio", "frequency_mode", "gallery_fold",
    "labels_available", "references_sha256", "reference_count", "query_batch_size", "runner_code_sha256",
    "v2_feature_code_sha256", "wand_code_sha256", "threshold", "partitions")
OUTPUT_NAMES = {"candidate_pairs.tsv", "matching_results.tsv", "owner_evidence.npz"}
STATE = None


def read_json(path):
    return json.loads(Path(path).read_text())


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def recipe(config):
    missing = set(RECIPE_KEYS) - set(config)
    if missing:
        raise ValueError(f"Missing v2 inference identity fields: {sorted(missing)}")
    return {k: config[k] for k in RECIPE_KEYS}


def load_release(path):
    release = read_json(path)
    if release.get("schema_version") != 1 or release.get("training_references") != 100000:
        raise ValueError("Expected frozen matched100k release")
    if release.get("final_audit_used_for_selection", True) or release.get("external_data_used", True):
        raise ValueError("Release must precede final audit and exclude external matching data")
    rule = release["owner_policy"]
    expected = {"eligibility": "all pairs with original model probability >= threshold; no route gate",
        "reject_when": "best_other_strength > own_strength + allowance", "text_strength_storage_dtype": "float32",
        "unknown_competitor_strength": -1., "same_reference_excluded": True, "arbitrary_tie_breaking": False,
        "channels": ["name_words", "ascii_words", "combined"], "reverse_top_k_per_channel": 10,
        "country_block": True, "max_df_ratio": .01, "frequency_mode": "gate", "label_free": True,
        "multiple_targets_per_reference_allowed": True, "unmatched_targets_allowed": True}
    if any(rule.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen owner semantics are not supported exactly")
    threshold, allowance = release["threshold"], rule["allowance"]
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Invalid frozen threshold")
    if rule["enabled"] != (allowance is not None) or (allowance is not None and not np.isfinite(allowance)):
        raise ValueError("Invalid frozen allowance/enable state")
    if release["artifacts"]["code_v2_owner_competition.py"]["sha256"] != file_hash(v2_owner_competition.__file__):
        raise ValueError("Owner implementation changed since release freeze")
    return release


def reverse_fingerprint(store, index, expected_references=None):
    store, index = Path(store), Path(index)
    config, manifest = read_json(index / "config.json"), read_json(index / "manifest.json")
    if (not manifest.get("complete") or not manifest.get("full_source1_coverage")
            or config.get("limit") is not None or config["store_sha256"] != file_hash(store / "config.json")):
        raise ValueError("Reverse index must cover all Source1 text in the supplied store")
    wal = store / "records.sqlite3-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Require immutable checkpointed store, not pending WAL writes")
    db = connect(store)
    actual = db.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
    db.close()
    idx = tantivy.Index.open(str(index / "tantivy"))
    if not (actual == manifest["indexed_records"] == manifest["expected_records"] == idx.searcher().num_docs):
        raise ValueError("Reverse-index/source full coverage mismatch")
    if expected_references is not None and actual != expected_references:
        raise ValueError("Production references must cover the complete test Source1 gallery")
    # Locks and Tantivy's mutable managed-file bookkeeping are not index content.
    files = {p.name: file_hash(p) for p in sorted((index / "tantivy").iterdir())
             if p.is_file() and not p.name.startswith(".")}
    if "meta.json" not in files:
        raise ValueError("Missing reverse index metadata")
    return {"config_sha256": file_hash(index / "config.json"), "manifest_sha256": file_hash(index / "manifest.json"),
        "tantivy_files": files, "content_sha256": json_hash(files), "source1_records": actual,
        "store_config_sha256": file_hash(store / "config.json"), "store_sqlite_sha256": file_hash(store / "records.sqlite3"),
        "component_sha256": {n: file_hash(importlib.import_module(n).__file__) for n in
                              ("ber.text", "ber.disk_index", "ber.retrieval", "v2_owner_competition")}}


def validate_raw_shard(folder, rows, config, config_sha):
    folder = Path(folder)
    done = read_json(folder / "done.json")
    expected = {"candidate_pairs.tsv", "matching_results.tsv", "probabilities.npy", "pair_ids.npz"}
    if config.get("emit_matrix"):
        expected.add("X76.npy")
    if set(done["output_sha256"]) != expected:
        raise ValueError("Unexpected raw inference file set")
    if (done["config_sha256"] != config_sha or done["input_sha256"] != signature(rows)
            or done["references"] != len(rows) or done["shard_id"] != int(folder.name)
            or done["reference_ids_sha256"] != hashlib.sha256("\n".join(r["entity_id"] for r in rows).encode()).hexdigest()):
        raise ValueError("Raw inference input/config/coverage mismatch")
    for name, digest in done["output_sha256"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Raw inference output checksum mismatch")
    p = np.load(folder / "probabilities.npy", allow_pickle=False)
    with np.load(folder / "pair_ids.npz", allow_pickle=False) as z:
        # Never load y/true_counts here, even for separately authorized audit inference.
        a = {k: z[k] for k in ("reference_ids", "target_ids", "query_ids", "countries")}
    ids = [r["entity_id"] for r in rows]
    if list(a["reference_ids"]) != ids or list(a["countries"]) != [r["country"] for r in rows]:
        raise ValueError("Raw pair/reference alignment mismatch")
    q = a["query_ids"]
    if (p.shape != (done["candidate_pairs"],) or q.shape != p.shape or a["target_ids"].shape != p.shape
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or q.dtype.kind not in "iu" or (len(q) and (q.min() < 0 or q.max() >= len(rows)))):
        raise ValueError("Invalid raw probability/query alignment")
    groups = [set() for _ in rows]
    for owner, target in zip(q, a["target_ids"]):
        if not str(target).startswith(("S2-", "S3-")) or str(target) in groups[int(owner)]:
            raise ValueError("Invalid/duplicate raw candidate")
        groups[int(owner)].add(str(target))
    for name, column, accepted in (("candidate_pairs.tsv", "candidate_entity_ids", None),
                                  ("matching_results.tsv", "matched_entity_ids", p >= config["threshold"])):
        records = list(read_tsv(folder / name))
        if len(records) != len(rows):
            raise ValueError("Raw TSV omitted empty reference rows")
        chosen = groups if accepted is None else [set() for _ in rows]
        if accepted is not None:
            for owner, target in zip(q[accepted], a["target_ids"][accepted]):
                chosen[int(owner)].add(str(target))
        for row, rid, expected_ids in zip(records, ids, chosen):
            values = row.get(column, "").split(",") if row.get(column) else []
            if set(row) != {"source1_entity_id", column} or row["source1_entity_id"] != rid or len(set(values)) != len(values) or set(values) != expected_ids:
                raise ValueError("Raw TSV differs from aligned probabilities/candidates")
    return done, a, p


def apply_rule(probabilities, own, other, threshold, allowance):
    own, other = np.asarray(own, dtype=np.float32), np.asarray(other, dtype=np.float32)
    if not np.isfinite(own).all() or not np.isfinite(other).all():
        raise ValueError("Nonfinite owner evidence")
    adjusted = probabilities if allowance is None else veto(probabilities, own, other, threshold, allowance)
    return adjusted >= threshold


def verify_output(folder, rows, config_sha, raw_done_sha):
    done = read_json(folder / "done.json")
    if (done["config_sha256"] != config_sha or done["raw_done_sha256"] != raw_done_sha
            or done["input_sha256"] != signature(rows) or done["references"] != len(rows)):
        raise ValueError("Postprocessor resume provenance/input mismatch")
    if set(done["output_sha256"]) != OUTPUT_NAMES:
        raise ValueError("Postprocessor output file set changed")
    for name, digest in done["output_sha256"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Postprocessor output checksum mismatch")
    return done


def initialize(config):
    global STATE
    competition = Competition(config["store"], config["reverse_index"], top_k=config["owner_policy"]["reverse_top_k_per_channel"])
    competition.db.execute("PRAGMA cache_size=-8192")
    STATE = config, competition


def process(task):
    shard, rows = task
    config, competition = STATE
    started = time.perf_counter()
    raw = Path(config["source"]) / "shards" / f"{shard:06d}"
    destination = Path(config["output"]) / "shards" / f"{shard:06d}"
    original, a, probabilities = validate_raw_shard(raw, rows, config["source_config"], config["source_config_sha256"])
    raw_hash, config_hash = file_hash(raw / "done.json"), file_hash(Path(config["output"]) / "config.json")
    if destination.exists():
        return verify_output(destination, rows, config_hash, raw_hash)
    ensure_disk(config["output"], config["minimum_free_gib"])
    ids = [r["entity_id"] for r in rows]
    anchors = lookup(competition.db, "anchors", "entity_id", ids)
    if len(anchors) != len(ids) or any(record_row(anchors[r["entity_id"]]) != r for r in rows):
        raise ValueError("Reference text differs from reverse-index store")
    references = [Record.from_row(row) for row in rows]
    selected = np.flatnonzero(probabilities >= config["threshold"])
    own = np.zeros(len(selected), dtype=np.float32)
    other = np.full(len(selected), -1., dtype=np.float32)
    rivals = np.zeros(len(selected), dtype=np.int16)
    found = np.zeros(len(selected), dtype=np.int8)
    # Existence and country checks are restricted to accepted pairs, as is veto work.
    targets = lookup(competition.db, "records", "entity_id", list(set(map(str, a["target_ids"][selected]))))
    if len(targets) != len(set(a["target_ids"][selected])):
        raise ValueError("Accepted target missing from original store")
    for j, i in enumerate(selected):
        reference = references[int(a["query_ids"][i])]
        target = str(a["target_ids"][i])
        if targets[target]["country"] != reference.country:
            raise ValueError("Country blocking violated")
        if config["owner_policy"]["enabled"]:
            own[j], other[j], rivals[j], found[j] = competition.feature(reference, target)
    keep = apply_rule(probabilities[selected], own, other, config["threshold"], config["owner_policy"]["allowance"])
    matches = [[] for _ in rows]
    for i in selected[keep]:
        matches[int(a["query_ids"][i])].append(str(a["target_ids"][i]))
    staging = Path(tempfile.mkdtemp(prefix=f".{shard:06d}.", dir=destination.parent))
    # Copy, not hardlink: modifying a downstream output must never alter raw inference.
    with (raw / "candidate_pairs.tsv").open("rb") as source, (staging / "candidate_pairs.tsv").open("xb") as target:
        shutil.copyfileobj(source, target)
    write_tsv(staging / "matching_results.tsv", ({"source1_entity_id": rid, "matched_entity_ids": ",".join(sorted(group))}
              for rid, group in zip(ids, matches)), ["source1_entity_id", "matched_entity_ids"])
    np.savez_compressed(staging / "owner_evidence.npz", pair_indices=selected.astype(np.int32), own=own, other=other,
        rivals=rivals, reverse_found=found, kept=keep, probabilities=probabilities[selected],
        query_ids=a["query_ids"][selected], target_ids=a["target_ids"][selected], reference_ids=a["reference_ids"], countries=a["countries"])
    done = {"shard_id": shard, "config_sha256": config_hash, "raw_done_sha256": raw_hash,
        "raw_output_sha256": original["output_sha256"], "input_sha256": original["input_sha256"],
        "reference_ids_sha256": original["reference_ids_sha256"], "references": len(rows),
        "candidate_pairs": len(probabilities), "threshold_accepted_pairs": len(selected), "kept_pairs": int(keep.sum()),
        "vetoed_pairs": int((~keep).sum()), "empty_predictions": sum(not group for group in matches),
        "wall_seconds": time.perf_counter()-started, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "output_sha256": {name: file_hash(staging / name) for name in sorted(OUTPUT_NAMES)}}
    if done["output_sha256"]["candidate_pairs.tsv"] != original["output_sha256"]["candidate_pairs.tsv"]:
        raise ValueError("Candidate output changed unexpectedly")
    write_json(staging / "done.json", done)
    if destination.exists():
        raise FileExistsError("Concurrent owner postprocessor committed same shard")
    staging.rename(destination)
    return done


def run(args):
    source, output, store, index = map(Path, (args.source, args.output, args.store, args.reverse_index))
    raw, release = read_json(source / "config.json"), load_release(args.release)
    identity = recipe(raw)
    if raw["mode"] != "predict" or raw["gallery_fold"] is not None:
        raise ValueError("Require full-gallery v2 predictions")
    if not args.training_diagnostic and (raw["labels_available"] or raw["store_split"] != "test"):
        raise ValueError("Production postprocessing requires unlabeled test inference")
    if args.training_diagnostic and raw["store_split"] != "train":
        raise ValueError("Training diagnostic requires the provided training store")
    if (raw["model_sha256"] != release["model_sha256"]
            or raw["model_report_sha256"] != release["artifacts"]["model_report"]["sha256"]
            or raw["training_manifest_sha256"] != release["artifacts"]["prepared_train_manifest"]["sha256"]
            or raw["v2_feature_code_sha256"] != release["artifacts"]["code_v2_cpu_features.py"]["sha256"]):
        raise ValueError("Inference is not the frozen released model/training/feature recipe")
    if file_hash(args.references) != raw["references_sha256"] or file_hash(store / "config.json") != raw["store_config_sha256"]:
        raise ValueError("Reference/store override differs from raw inference")
    fingerprint = reverse_fingerprint(store, index, None if args.training_diagnostic else raw["reference_count"])
    config = {"mode": "owner_predict", "source": str(source.resolve()), "source_config_sha256": file_hash(source / "config.json"),
        "source_config": raw, "inference_recipe": identity, "store": str(store.resolve()), "reverse_index": str(index.resolve()),
        "reverse_fingerprint": fingerprint, "references": str(Path(args.references).resolve()),
        "release_sha256": file_hash(args.release), "model_sha256": release["model_sha256"], "threshold": release["threshold"],
        "owner_policy": release["owner_policy"], "runner_sha256": file_hash(__file__), "output": str(output.resolve()),
        "training_diagnostic": args.training_diagnostic, "labels_read": False, "minimum_free_gib": args.minimum_free_gib,
        "partition": raw["partition"], "partitions": raw["partitions"], "assigned_partitions": raw["assigned_partitions"]}
    if raw["assigned_partitions"] != [raw["partition"]]:
        raise ValueError("Only a single declared raw partition per source is supported")
    if output.exists():
        if not args.resume or read_json(output / "config.json") != config:
            raise ValueError("Identical source/release/index/code configuration and --resume required")
    else:
        (output / "shards").mkdir(parents=True)
        write_json(output / "config.json", config)
    total = math.ceil(raw["reference_count"] / raw["query_batch_size"])
    expected = {i for i in range(total) if i % raw["partitions"] == raw["partition"]}
    committed = {int(p.name) for p in (source / "shards").iterdir() if p.name.isdecimal() and (p / "done.json").exists()}
    existing = {int(p.name) for p in (output / "shards").iterdir() if p.name.isdecimal()}
    if not committed <= expected or not existing <= committed:
        raise ValueError("Unexpected/missing raw or postprocessed committed shard")
    if not args.allow_incomplete and committed != expected:
        raise ValueError("Raw partition incomplete; explicit --allow-incomplete required for streaming")
    started = time.perf_counter()
    counts = {"references": 0, "candidate_pairs": 0, "kept_pairs": 0, "vetoed_pairs": 0, "empty_predictions": 0}
    inventory = []
    traversed = 0

    def tasks():
        nonlocal traversed
        new = 0
        for i, rows in enumerate(batches(read_tsv(args.references), raw["query_batch_size"])):
            traversed += len(rows)
            if i not in committed:
                continue
            if args.stop_after_shards and i not in existing and new >= args.stop_after_shards:
                continue
            if i not in existing:
                new += 1
            yield i, rows

    if args.workers == 1:
        initialize(config)
        results, pool = map(process, tasks()), None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config,),
                                   mp_context=multiprocessing.get_context("spawn"))
        # Bounded tasks prevent eager loading the full reference corpus.
        def dispatch():
            pending = []
            for task in tasks():
                pending.append(pool.submit(process, task))
                if len(pending) >= args.workers * 2:
                    yield pending.pop(0).result()
            for future in pending:
                yield future.result()
        results = dispatch()
    try:
        for done in results:
            for key in counts:
                counts[key] += done[key]
            inventory.append({"shard_id": done["shard_id"], "done_sha256": file_hash(output / "shards" / f"{done['shard_id']:06d}" / "done.json")})
            progress = {**counts, "completed_shards": len(inventory), "expected_shards": len(expected), "seconds": time.perf_counter()-started}
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    finally:
        if pool:
            pool.shutdown()
        elif STATE is not None:
            STATE[1].db.close()
    if traversed != raw["reference_count"] or file_hash(args.references) != raw["references_sha256"]:
        raise ValueError("Reference coverage/file changed")
    if file_hash(source / "config.json") != config["source_config_sha256"]:
        raise ValueError("Raw config changed during postprocessing")
    complete = {i["shard_id"] for i in inventory} == expected
    report = {"complete": complete, "partition_complete": complete, "complete_submission": False, **counts,
        "partition": raw["partition"], "partitions": raw["partitions"], "shards": inventory,
        "config_sha256": file_hash(output / "config.json"), "release_sha256": config["release_sha256"],
        "seconds": time.perf_counter()-started, "labels_read": False}
    write_json(output / "manifest.json", report)
    print(json.dumps(report), flush=True)


def audit(args):
    source, post, output = Path(args.source), Path(args.postprocessed), Path(args.output)
    release = load_release(args.release)
    config, manifest = read_json(post / "config.json"), read_json(post / "manifest.json")
    raw = read_json(source / "config.json")
    if (not config["training_diagnostic"] or not raw["labels_available"] or raw["store_split"] != "train"
            or raw["partitions"] != 1 or not manifest["complete"] or not read_json(source / "manifest.json").get("complete")):
        raise ValueError("Frozen audit requires complete single-partition labeled training inference")
    if (config["release_sha256"] != file_hash(args.release) or config["source_config_sha256"] != file_hash(source / "config.json")
            or manifest["config_sha256"] != file_hash(post / "config.json") or release["model_sha256"] != raw["model_sha256"]):
        raise ValueError("Audit release/source provenance mismatch")
    chunks = {k: [] for k in ("query_ids", "y", "p", "after", "true_counts", "countries")}
    offset = 0
    for shard in manifest["shards"]:
        folder = post / "shards" / f"{shard['shard_id']:06d}"
        original = source / "shards" / folder.name
        done = read_json(folder / "done.json")
        if (file_hash(folder / "done.json") != shard["done_sha256"] or file_hash(original / "done.json") != done["raw_done_sha256"]):
            raise ValueError("Audit shard commit changed")
        for root, files in ((folder, done["output_sha256"]), (original, done["raw_output_sha256"])):
            for name, digest in files.items():
                if file_hash(root / name) != digest:
                    raise ValueError("Audit committed file changed")
        with np.load(original / "pair_ids.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in ("query_ids", "y", "true_counts", "countries", "reference_ids")}
        p = np.load(original / "probabilities.npy", allow_pickle=False)
        after = np.full(len(p), -1., dtype=p.dtype)
        with np.load(folder / "owner_evidence.npz", allow_pickle=False) as z:
            indices = z["pair_indices"]
            if not np.array_equal(indices, np.flatnonzero(p >= release["threshold"])):
                raise ValueError("Audit owner evidence eligibility differs")
            expected = apply_rule(p[indices], z["own"], z["other"], release["threshold"], release["owner_policy"]["allowance"])
            if not np.array_equal(expected, z["kept"]):
                raise ValueError("Audit frozen owner decisions differ")
            after[indices[expected]] = p[indices[expected]]
        if not np.isin(a["y"], [0, 1]).all() or (a["true_counts"] < 0).any():
            raise ValueError("Invalid provided audit labels")
        for key in ("y", "true_counts", "countries"):
            chunks[key].append(a[key])
        chunks["query_ids"].append(a["query_ids"] + offset)
        chunks["p"].append(p); chunks["after"].append(after)
        offset += len(a["reference_ids"])
    if offset != raw["reference_count"]:
        raise ValueError("Audit coverage incomplete")
    a = {k: np.concatenate(v) for k, v in chunks.items()}
    baseline_threshold = release["comparison"]["tune"]["policies"]["fixed_baseline"]["policy"]["threshold"]
    reports, values = {}, {}
    for label, p, threshold in (("cpu_baseline", a["p"], baseline_threshold),
                                ("threshold_only", a["p"], release["threshold"]), ("frozen_owner", a["after"], release["threshold"])):
        result = score_arrays(a["query_ids"], a["y"], p, a["true_counts"], threshold)
        scores = result.pop("entity_scores")
        result["countries"] = {str(c): {"references": int((a["countries"] == c).sum()),
            "macro_f05": float(scores[a["countries"] == c].mean())} for c in np.unique(a["countries"])}
        reports[label], values[label] = result, scores
    delta = values["frozen_owner"] - values["cpu_baseline"]
    rng = np.random.default_rng(20260925)
    boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(3000)]
    output.mkdir(parents=True, exist_ok=False)
    report = {"complete": True, "references": offset, "candidate_pairs": len(a["p"]), "release_sha256": file_hash(args.release),
        "inference_manifest_sha256": file_hash(source / "manifest.json"), "owner_manifest_sha256": file_hash(post / "manifest.json"),
        "threshold": release["threshold"], "allowance": release["owner_policy"]["allowance"], "metrics": reports,
        "delta_vs_cpu_baseline": float(delta.mean()), "paired_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
        "policy_tuned_on_audit": False, "note": "One frozen-policy audit; no threshold/grid selection performed."}
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run")
    for name in ("source", "references", "store", "reverse-index", "release", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--minimum-free-gib", type=float, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--training-diagnostic", action="store_true")
    p.add_argument("--stop-after-shards", type=int)
    p.set_defaults(fn=run)
    p = commands.add_parser("audit")
    for name in ("source", "postprocessed", "release", "output"):
        p.add_argument("--" + name, required=True)
    p.set_defaults(fn=audit)
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1 or getattr(args, "minimum_free_gib", 6) < 0 or (getattr(args, "stop_after_shards", None) is not None and args.stop_after_shards < 1):
        parser.error("Invalid worker/disk/shard limit")
    args.fn(args)
