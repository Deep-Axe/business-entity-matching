"""Shared read-only integrity checks for released v2 raw CPU predictions."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ber.data import read_tsv
from ber.shards import file_hash
from v2_run_pipeline import NAMES, WEIGHTING_POLICY, signature, code_hashes
from ber.direct_features import CHANNELS
import v2_run_pipeline


RECIPE_KEYS = ("mode", "model_sha256", "model_report_sha256", "training_manifest_sha256",
    "feature_names", "component_sha256", "weighting_policy", "weighting_index_config_sha256",
    "weighting_target_records", "training_weighting_index_config_sha256", "store_config_sha256",
    "store_split", "index_config_sha256", "lexicon_sha256", "gate_terms", "top_k_per_channel",
    "channels", "channel_order", "country_block", "max_df_ratio", "frequency_mode", "gallery_fold",
    "labels_available", "references_sha256", "reference_count", "query_batch_size", "runner_code_sha256",
    "v2_feature_code_sha256", "wand_code_sha256", "threshold", "partitions")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_200k_release(path):
    release = read_json(path)
    if (release.get("schema_version") != 1 or release.get("training_references") != 200000
            or release.get("final_audit_used_for_selection", True) or release.get("external_data_used", True)):
        raise ValueError("Expected frozen audit-free provided-data200k CPU release")
    if release["owner_policy"].get("enabled") is not False or release["owner_policy"].get("allowance") is not None:
        raise ValueError("This assembler/evaluator supports only the frozen CPU-only release")
    if not np.isfinite(release["threshold"]) or not 0 < release["threshold"] <= 1:
        raise ValueError("Invalid frozen threshold")
    if release.get('threshold_selection') != 'full_tune only':
        raise ValueError('200k threshold must be selected exclusively on full tuning')
    if release.get('expected_test_references') != 1732544:
        raise ValueError('Expected complete competition test scope')
    return release


def identity(config):
    missing = set(RECIPE_KEYS) - set(config)
    if missing:
        raise ValueError(f"Missing v2 inference identity fields: {sorted(missing)}")
    return {key: config[key] for key in RECIPE_KEYS}


def validate_config(config, release, audit):
    if config["mode"] != "predict" or config["gallery_fold"] is not None or config["labels_available"] != audit:
        raise ValueError("Wrong prediction/label/full-gallery mode")
    if config["store_split"] != ("train" if audit else "test"):
        raise ValueError("Wrong source store split")
    if (config["model_sha256"] != release["model_sha256"] or config["threshold"] != release["threshold"]
            or config["model_report_sha256"] != release["artifacts"]["model_report"]["sha256"]
            or config["training_manifest_sha256"] != release["artifacts"]["prepared_train_manifest"]["sha256"]
            or config["v2_feature_code_sha256"] != release["artifacts"]["code_v2_cpu_features.py"]["sha256"]):
        raise ValueError("Prediction differs from frozen model/threshold/training/feature release")
    if (config["feature_names"] != NAMES or config["weighting_policy"] != WEIGHTING_POLICY
            or config["gate_terms"] != 0 or config["top_k_per_channel"] != 20
            or not config["country_block"] or config["weighting_index_config_sha256"] != config["index_config_sha256"]
            or config["channels"] != list(CHANNELS) or config["channel_order"] != list(CHANNELS)
            or config["max_df_ratio"] != .01 or config["frequency_mode"] != "gate"):
        raise ValueError("Expected matched gate0/76-feature/forward-gallery weighting recipe")
    if config["component_sha256"].get("v2_features") != config["v2_feature_code_sha256"]:
        raise ValueError("Contradictory enhanced-feature implementation identity")
    if config.get("external_data_used", True):
        raise ValueError("External matching data is not part of this release")
    if config["component_sha256"] != code_hashes() or config["runner_code_sha256"] != file_hash(v2_run_pipeline.__file__):
        raise ValueError("Inference feature/runner code differs from checked implementation")


def inventory(sources, references, release_path, audit=False):
    release = read_200k_release(release_path)
    reference_hash = file_hash(references)
    located, sources_info, partitions = {}, [], set()
    common = None
    for value in sources:
        source = Path(value)
        config = read_json(source / "config.json")
        current = identity(config)
        validate_config(config, release, audit)
        if common is None:
            common = current
        elif current != common:
            differences = [k for k in RECIPE_KEYS if current[k] != common[k]]
            raise ValueError(f"Partition recipe mismatch: {differences}")
        if config["references_sha256"] != reference_hash:
            raise ValueError("Unexpected full reference TSV")
        partition = config["partition"]
        if (not 0 <= partition < config["partitions"] or config["assigned_partitions"] != [partition]
                or partition in partitions):
            raise ValueError("Invalid/duplicate declared partition")
        partitions.add(partition)
        manifest = read_json(source / "manifest.json")
        if (not manifest.get("complete") or not manifest.get("partition_complete")
                or manifest.get("config") != config or manifest.get("partition") != partition
                or manifest.get("partitions") != config["partitions"]):
            raise ValueError("Raw partition has no consistent complete manifest")
        count, size = config["reference_count"], config["query_batch_size"]
        if min(count, size, config["partitions"]) < 1:
            raise ValueError("Invalid partition/reference sizes")
        expected = {i for i in range(math.ceil(count / size)) if i % config["partitions"] == partition}
        actual = {int(p.name) for p in (source / "shards").iterdir() if p.name.isdecimal()}
        if actual != expected:
            raise ValueError("Partition missing/extra numeric shard directories")
        expected_refs = sum(min(size, count-i*size) for i in expected)
        if manifest["references"] != expected_refs:
            raise ValueError("Partition manifest reference count mismatch")
        item = {"path": str(source.resolve()), "config": config, "config_sha256": file_hash(source / "config.json"),
                "manifest_sha256": file_hash(source / "manifest.json"), "partition": partition,
                "declared_candidate_pairs": manifest["candidate_pairs"], "verified_candidate_pairs": 0}
        sources_info.append(item)
        for shard in expected:
            if shard in located:
                raise ValueError("Duplicate global shard; no implicit deduplication")
            folder = source / "shards" / f"{shard:06d}"
            if not (folder / "done.json").is_file():
                raise ValueError("Missing committed shard marker")
            located[shard] = folder, item
    if common is None:
        raise ValueError("No prediction sources")
    if partitions != set(range(common["partitions"])):
        raise ValueError("Not all declared partitions are present")
    if set(located) != set(range(math.ceil(common["reference_count"] / common["query_batch_size"]))):
        raise ValueError("Incomplete global shard coverage")
    return release, common, located, sources_info


def validate_shard(folder, rows, config, config_sha):
    done = read_json(folder / "done.json")
    expected_files = {"candidate_pairs.tsv", "matching_results.tsv", "probabilities.npy", "pair_ids.npz"}
    if config.get("emit_matrix"):
        expected_files.add("X76.npy")
    if set(done["output_sha256"]) != expected_files:
        raise ValueError("Unexpected inference file set")
    if (done["config_sha256"] != config_sha or done["input_sha256"] != signature(rows)
            or done["references"] != len(rows) or done["shard_id"] != int(folder.name)
            or done["reference_ids_sha256"] != hashlib.sha256("\n".join(r["entity_id"] for r in rows).encode()).hexdigest()):
        raise ValueError("Shard input/config/coverage mismatch")
    for name, digest in done["output_sha256"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Inference output checksum mismatch")
    p = np.load(folder / "probabilities.npy", allow_pickle=False)
    with np.load(folder / "pair_ids.npz", allow_pickle=False) as z:
        # Assembly never reads labels, even if a malicious file contains them.
        a = {k: z[k] for k in ("reference_ids", "target_ids", "query_ids", "countries")}
    ids = [r["entity_id"] for r in rows]
    if list(a["reference_ids"]) != ids or list(a["countries"]) != [r["country"] for r in rows]:
        raise ValueError("Pair/reference alignment mismatch")
    q = a["query_ids"]
    if (p.shape != (done["candidate_pairs"],) or q.shape != p.shape or a["target_ids"].shape != p.shape
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or q.dtype.kind not in "iu" or (len(q) and (q.min() < 0 or q.max() >= len(rows)))):
        raise ValueError("Invalid probability/query alignment")
    candidates, matches = [set() for _ in rows], [set() for _ in rows]
    for owner, target, probability in zip(q, a["target_ids"], p):
        if not str(target).startswith(("S2-", "S3-")) or str(target) in candidates[int(owner)]:
            raise ValueError("Invalid/duplicate candidate pair")
        candidates[int(owner)].add(str(target))
        if probability >= config["threshold"]:
            matches[int(owner)].add(str(target))
    for name, column, groups in (("candidate_pairs.tsv", "candidate_entity_ids", candidates),
                                 ("matching_results.tsv", "matched_entity_ids", matches)):
        with (folder / name).open(encoding="utf-8") as handle:
            if next(handle).rstrip("\r\n") != "source1_entity_id\t" + column:
                raise ValueError("Unexpected TSV column order/schema")
        records = list(read_tsv(folder / name))
        if len(records) != len(rows):
            raise ValueError("TSV omitted reference rows including empties")
        for row, rid, expected_ids in zip(records, ids, groups):
            values = row.get(column, "").split(",") if row.get(column) else []
            if set(row) != {"source1_entity_id", column} or row["source1_entity_id"] != rid or len(set(values)) != len(values) or set(values) != expected_ids:
                raise ValueError("TSV differs from aligned probabilities/candidates")
    return done, a, p, {"predicted_pairs": sum(map(len, matches)), "empty_predictions": sum(not v for v in matches)}


def verify_source_totals(sources):
    for item in sources:
        if item["verified_candidate_pairs"] != item["declared_candidate_pairs"]:
            raise ValueError("Partition manifest candidate count differs from verified shards")
        root = Path(item["path"])
        if file_hash(root / "config.json") != item["config_sha256"] or file_hash(root / "manifest.json") != item["manifest_sha256"]:
            raise ValueError("Partition config/manifest changed during verification")
