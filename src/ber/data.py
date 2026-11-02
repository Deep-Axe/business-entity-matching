"""Streaming, entity-disjoint local development samples."""

import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

FIELDS = ["entity_id", "business_name", "business_address", "country"]
GOLD_FIELDS = ["source1_entity_id", "matched_entity_ids"]
FOLDS = ("train", "tune", "eval")


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    # Commit generated metadata atomically so interruption cannot leave a
    # half-written manifest. Uncommitted temporary files are not removed here.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def fold_for(entity_id, seed):
    digest = hashlib.blake2b(f"{seed}:{entity_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % 10000
    return "train" if bucket < 7000 else "tune" if bucket < 8500 else "eval"


def reservoir_add(items, item, seen, capacity, rng):
    if len(items) < capacity:
        items.append(item)
    elif capacity:
        slot = rng.randrange(seen)
        if slot < capacity:
            items[slot] = item


def prepare_sample(data_dir, output, train_size, tune_size, eval_size,
                   distractors=50000, seed=2026):
    """Keep all selected positives; assign extra records by their gold owner.

    Galleries are disjoint across train/tune/eval. Even sampled distractors with
    known owners follow the owner's fold. No challenge test labels are used.
    """
    if min(train_size, tune_size, eval_size) <= 0 or distractors < 0:
        raise ValueError("All fold sizes must be positive; distractors nonnegative.")
    data_dir, output = Path(data_dir), Path(output)
    required = [data_dir / "train" / f"train_source{i}.tsv" for i in (1, 2, 3)]
    gold_path = data_dir / "train/train_ground_truth.tsv"
    for path in [*required, gold_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    capacities = dict(zip(FOLDS, (train_size, tune_size, eval_size)))
    samples, seen = {f: [] for f in FOLDS}, Counter()
    rng = random.Random(seed)
    for row in read_tsv(gold_path):
        fold = fold_for(row["source1_entity_id"], seed)
        seen[fold] += 1
        reservoir_add(samples[fold], row, seen[fold], capacities[fold], rng)
    if any(len(samples[f]) < capacities[f] for f in FOLDS):
        raise ValueError("Not enough entities in one or more folds.")
    anchors, wanted, expected_owner = {}, {}, {}
    for fold, rows in samples.items():
        for row in rows:
            anchor_id = row["source1_entity_id"]
            if anchor_id in anchors:
                raise ValueError(f"Duplicate sampled anchor {anchor_id}")
            anchors[anchor_id] = fold
            matches = [x for x in row["matched_entity_ids"].split(",") if x]
            if len(matches) != len(set(matches)):
                raise ValueError(f"Duplicate match in ground truth for {anchor_id}")
            for target_id in matches:
                if target_id in expected_owner:
                    raise ValueError(f"Multiple sampled owners for {target_id}")
                wanted[target_id] = fold
                expected_owner[target_id] = anchor_id
    print(f"Selected {len(anchors):,} reference groups, "
          f"{len(wanted):,} known positive targets.", flush=True)
    references = {f: [] for f in FOLDS}
    found_anchors = set()
    for row in read_tsv(required[0]):
        fold = anchors.get(row["entity_id"])
        if fold is not None:
            if row["entity_id"] in found_anchors:
                raise ValueError("Duplicate selected S1 ID in source data.")
            found_anchors.add(row["entity_id"])
            references[fold].append(row)
    if found_anchors != set(anchors):
        raise ValueError("Some sampled S1 IDs are absent from source data.")
    targets = {f: [] for f in FOLDS}
    extra, extra_seen, found = [], 0, set()
    for path in required[1:]:
        for row in read_tsv(path):
            target_id = row["entity_id"]
            if target_id in wanted:
                if target_id in found:
                    raise ValueError("Duplicate selected target ID.")
                found.add(target_id)
                targets[wanted[target_id]].append(row)
            else:
                extra_seen += 1
                reservoir_add(extra, row, extra_seen, distractors, rng)
        print(f"Scanned {path.name}; retained positives and distractor reservoir.",
              flush=True)
    if found != set(wanted):
        raise ValueError("Some ground-truth targets are absent from source files.")
    extras_by_id = {row["entity_id"]: row for row in extra}
    if len(extras_by_id) != len(extra):
        raise ValueError("Duplicate IDs in distractor sample.")
    # Resolve owners for extras and audit all retained positive targets against
    # the entire label file, not just the selected anchor rows.
    checked_ids = set(extras_by_id) | set(wanted)
    owners = {}
    for row in read_tsv(gold_path):
        anchor_id = row["source1_entity_id"]
        for target_id in filter(None, row["matched_entity_ids"].split(",")):
            if target_id in checked_ids:
                previous = owners.setdefault(target_id, anchor_id)
                if previous != anchor_id:
                    raise ValueError(f"Ambiguous gold ownership for {target_id}; "
                                     "group connected anchors before splitting.")
    for target_id, anchor_id in expected_owner.items():
        if owners.get(target_id) != anchor_id:
            raise ValueError("Positive target ownership mismatch.")
    for target_id, row in extras_by_id.items():
        targets[fold_for(owners.get(target_id, target_id), seed)].append(row)
    report = {
        "seed": seed, "input_data_dir": str(data_dir.resolve()),
        "source": "provided training data only",
        "selection": "uniform reservoir within deterministic entity partitions",
        "fold_assignment": "blake2b(seed:owner_id), 70%/15%/15%",
        "distractors": len(extra),
        "distractors_with_known_owner": sum(x in owners for x in extras_by_id),
        "limitations": ["Sampled target galleries are smaller and easier than full galleries.",
                       "No France labels are available."],
        "folds": {},
    }
    for fold in FOLDS:
        folder = output / fold
        folder.mkdir()
        ref = sorted(references[fold], key=lambda r: r["entity_id"])
        target = sorted(targets[fold], key=lambda r: r["entity_id"])
        # test_source filenames permit the official validator to check each
        # local fold; all content still comes from the original training files.
        write_tsv(folder / "test_source1.tsv", ref, FIELDS)
        for source in (2, 3):
            write_tsv(folder / f"test_source{source}.tsv",
                      (r for r in target if r["entity_id"].startswith(f"S{source}-")),
                      FIELDS)
        write_tsv(folder / "ground_truth.tsv",
                  sorted(samples[fold], key=lambda r: r["source1_entity_id"]),
                  GOLD_FIELDS)
        report["folds"][fold] = {
            "references": len(ref), "targets": len(target),
            "countries": dict(Counter(r["country"] for r in ref)),
            "positive_pairs": sum(len(r["matched_entity_ids"].split(","))
                                  if r["matched_entity_ids"] else 0
                                  for r in samples[fold]),
            "singletons": sum(not r["matched_entity_ids"] for r in samples[fold]),
        }
    report["seconds"] = round(time.perf_counter() - started, 3)
    write_json(output / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
