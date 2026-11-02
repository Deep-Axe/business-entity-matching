"""Nested training expansion and a fresh audit, excluding all previous evaluated owners."""

import argparse
import json
import random
from pathlib import Path

from ber.data import FIELDS, read_tsv, reservoir_add, write_json, write_tsv
from ber.shards import file_hash, gold_for
from ber.store import connect, record_row


def run(store, previous, output, train_size=50000, audit_size=10000):
    output, previous = Path(output), Path(previous)
    output.mkdir(parents=True, exist_ok=False)
    existing = list(read_tsv(previous / "train/references.tsv"))
    if train_size <= len(existing):
        raise ValueError("Expansion must exceed prior training size")
    excluded = {r["entity_id"] for path in previous.glob("*/references.tsv") for r in read_tsv(path)}
    for path in Path("artifacts/local_sample_v1").glob("*/test_source1.tsv"):
        excluded.update(r["entity_id"] for r in read_tsv(path))
    rng = random.Random(2042)
    sampled = {"train": [], "eval": []}
    seen = {"train": 0, "eval": 0}
    capacity = {"train": train_size-len(existing), "eval": audit_size}
    db = connect(store)
    for row in db.execute("SELECT * FROM anchors ORDER BY rowid"):
        fold = row["fold"]
        if fold not in sampled or row["entity_id"] in excluded:
            continue
        seen[fold] += 1
        reservoir_add(sampled[fold], record_row(row), seen[fold], capacity[fold], rng)
    sampled["train"] = existing + sampled["train"]
    sampled["audit"] = sampled.pop("eval")
    report = dict(seed=2042, old_training_included=len(existing), excluded_previous_ids=len(excluded),
                  audit_policy="Freeze all choices before inspecting new audit scores", sets={})
    for name, rows in sampled.items():
        directory = output / name
        directory.mkdir()
        rows.sort(key=lambda row: row["entity_id"])
        gold = gold_for(db, [r["entity_id"] for r in rows])
        write_tsv(directory / "references.tsv", rows, FIELDS)
        write_tsv(directory / "ground_truth.tsv", (
            {"source1_entity_id": r["entity_id"], "matched_entity_ids": ",".join(sorted(gold[r["entity_id"]]))}
            for r in rows), ["source1_entity_id", "matched_entity_ids"])
        report["sets"][name] = dict(references=len(rows), positive_pairs=sum(map(len, gold.values())),
                                    references_sha256=file_hash(directory / "references.tsv"))
    db.close()
    write_json(output / "manifest.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "previous", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--audit-size", type=int, default=10000)
    run(**vars(parser.parse_args()))
