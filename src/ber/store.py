"""Resumable streaming ingestion and bounded-memory access to the provided data."""

import itertools
import json
import resource
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .data import fold_for, read_tsv, write_json

STORE_VERSION = 1


def batches(iterator, size):
    iterator = iter(iterator)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def ensure_disk(directory, minimum_gib=10):
    if shutil.disk_usage(directory).free < minimum_gib * 1024**3:
        raise RuntimeError(f"Less than {minimum_gib} GiB free disk; stopping safely.")


def connect(store, readonly=True):
    path = Path(store) / "records.sqlite3"
    if readonly:
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        db.execute("PRAGMA query_only=ON")
    else:
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA temp_store=FILE")
    return db


def lookup(db, table, key, values):
    # Table and key names are internal constants, never user-provided SQL.
    result = {}
    for group in batches(values, 800):
        sql = f"SELECT * FROM {table} WHERE {key} IN ({','.join('?' for _ in group)})"
        result.update((row[key], row) for row in db.execute(sql, group))
    return result


def record_row(row):
    return {"entity_id": row["entity_id"], "country": row["country"],
            "business_name": row["name"], "business_address": row["address"]}


def _validate(row, source):
    if not row["entity_id"].startswith(f"S{source}-"):
        raise ValueError(f"Unexpected source ID {row['entity_id']}")
    if not row["business_name"] or not row["country"] or None in row:
        raise ValueError(f"Malformed record {row['entity_id']}")


def build_store(data_dir, output, split="train", seed=2026, batch_size=25000,
                resume=False, stop_after_batches=None, minimum_free_gib=10):
    if split not in ("train", "test") or batch_size < 1:
        raise ValueError("Invalid split or batch size")
    output, data_dir = Path(output), Path(data_dir)
    inputs = [data_dir / split / f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    if split == "train":
        inputs.append(data_dir / split / "train_ground_truth.tsv")
    config = {
        "store_version": STORE_VERSION, "split": split, "seed": seed,
        "sources": {str(p.resolve()): {"bytes": p.stat().st_size,
                     "mtime_ns": p.stat().st_mtime_ns} for p in inputs},
        "external_data_used": False,
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Store configuration or input files changed; use a new output")
    else:
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
    ensure_disk(output, minimum_free_gib)
    started = time.perf_counter()
    db = connect(output, readonly=False)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS anchors (
            entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, fold TEXT NOT NULL,
            name TEXT NOT NULL, address TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS gold_rows (
            reference_id TEXT PRIMARY KEY, n_matches INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS labels (
            target_id TEXT PRIMARY KEY, reference_id TEXT NOT NULL,
            fold TEXT NOT NULL, country TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records (
            row_id INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL,
            source INTEGER NOT NULL, country TEXT NOT NULL, fold TEXT NOT NULL,
            name TEXT NOT NULL, address TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS progress (
            stage TEXT PRIMARY KEY, rows_done INTEGER NOT NULL, complete INTEGER NOT NULL,
            known_targets INTEGER NOT NULL DEFAULT 0,
            country_mismatches INTEGER NOT NULL DEFAULT 0);
    """)
    stages = [(f"{split}_source1", inputs[0], 1)]
    if split == "train":
        stages.append(("ground_truth", inputs[3], 0))
    stages.extend((f"{split}_source{i}", inputs[i - 1], i) for i in (2, 3))
    completed_batches = 0
    try:
        for stage, path, source in stages:
            previous = db.execute("SELECT * FROM progress WHERE stage=?", (stage,)).fetchone()
            if previous and previous["complete"]:
                print(f"{stage}: already complete.", flush=True)
                continue
            done = previous["rows_done"] if previous else 0
            known = previous["known_targets"] if previous else 0
            mismatches = previous["country_mismatches"] if previous else 0
            iterator = itertools.islice(read_tsv(path), done, None)
            for batch in batches(iterator, batch_size):
                ensure_disk(output, minimum_free_gib)
                with db:
                    if source == 1:
                        values = []
                        for row in batch:
                            _validate(row, 1)
                            values.append((row["entity_id"], row["country"],
                                           fold_for(row["entity_id"], seed) if split == "train" else "test",
                                           row["business_name"], row["business_address"]))
                        db.executemany("INSERT INTO anchors VALUES (?,?,?,?,?)", values)
                    elif source == 0:
                        anchors = lookup(db, "anchors", "entity_id",
                                         [r["source1_entity_id"] for r in batch])
                        labels, gold_rows = [], []
                        for row in batch:
                            eid = row["source1_entity_id"]
                            if eid not in anchors:
                                raise ValueError(f"Unknown labeled reference {eid}")
                            targets = list(filter(None, row["matched_entity_ids"].split(",")))
                            if len(targets) != len(set(targets)):
                                raise ValueError(f"Duplicate positive for {eid}")
                            if any(not t.startswith(("S2-", "S3-")) for t in targets):
                                raise ValueError(f"Invalid target ID in labels for {eid}")
                            gold_rows.append((eid, len(targets)))
                            labels.extend((t, eid, anchors[eid]["fold"], anchors[eid]["country"])
                                          for t in targets)
                        db.executemany("INSERT INTO gold_rows VALUES (?,?)", gold_rows)
                        # A target with two owners fails the unique constraint.
                        db.executemany("INSERT INTO labels VALUES (?,?,?,?)", labels)
                    else:
                        ownership = lookup(db, "labels", "target_id",
                                           [r["entity_id"] for r in batch]) if split == "train" else {}
                        values = []
                        for row in batch:
                            _validate(row, source)
                            eid = row["entity_id"]
                            owner = ownership.get(eid)
                            fold = owner["fold"] if owner else (
                                fold_for(eid, seed) if split == "train" else "test")
                            known += owner is not None
                            mismatches += owner is not None and owner["country"] != row["country"]
                            values.append((eid, source, row["country"], fold,
                                           row["business_name"], row["business_address"]))
                        db.executemany("""INSERT INTO records
                            (entity_id,source,country,fold,name,address) VALUES (?,?,?,?,?,?)""", values)
                    done += len(batch)
                    db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,0,?,?)",
                               (stage, done, known, mismatches))
                completed_batches += 1
                if done % 250000 < batch_size:
                    print(f"{stage}: {done:,} rows committed; "
                          f"{time.perf_counter() - started:.1f}s this invocation.", flush=True)
                if stop_after_batches and completed_batches >= stop_after_batches:
                    return {"complete": False, "stage": stage, "rows_done": done}
            with db:
                db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,1,?,?)",
                           (stage, done, known, mismatches))
            print(f"{stage}: finished {done:,} rows.", flush=True)
        print("Creating query indexes and checking full label coverage...", flush=True)
        db.executescript("""
            CREATE INDEX IF NOT EXISTS labels_reference ON labels(reference_id);
            CREATE INDEX IF NOT EXISTS anchors_fold ON anchors(fold);
        """)
        missing_targets = db.execute("""SELECT COUNT(*) FROM labels l LEFT JOIN
            records r ON r.entity_id=l.target_id WHERE r.row_id IS NULL""").fetchone()[0]
        missing_gold = db.execute("""SELECT COUNT(*) FROM anchors a LEFT JOIN
            gold_rows g ON g.reference_id=a.entity_id WHERE g.reference_id IS NULL""").fetchone()[0]
        if missing_targets or (split == "train" and missing_gold):
            raise ValueError(f"Missing targets={missing_targets}; unlabeled references={missing_gold}")
        report = {
            "complete": True, "config": config,
            "references": db.execute("SELECT COUNT(*) FROM anchors").fetchone()[0],
            "targets": db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
            "positive_pairs": db.execute("SELECT COUNT(*) FROM labels").fetchone()[0],
            "targets_by_fold_country": [dict(r) for r in db.execute(
                "SELECT fold,country,COUNT(*) AS records FROM records GROUP BY fold,country")],
            "country_disagreement_positive_pairs": db.execute(
                "SELECT SUM(country_mismatches) FROM progress").fetchone()[0],
            "checks": {"unique_reference_ids": True, "unique_target_ids": True,
                       "unique_gold_reference_rows": True, "at_most_one_owner_per_target": True,
                       "all_labeled_targets_exist": True,
                       "all_references_labeled": split == "train"},
            "last_invocation_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        write_json(output / "manifest.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        db.close()
