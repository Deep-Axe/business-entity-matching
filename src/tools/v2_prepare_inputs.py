"""Make nested training selections and a never-scored fresh audit reference set.

No ground-truth file/table is read. Existing 50k training references are an exact
ordered prefix of both larger selections. Existing tuning/development files are
copied byte-for-byte; all historic selections are excluded from new sampling.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import time

from ber.data import FIELDS, read_tsv, write_json, write_tsv
from ber.store import connect, lookup


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def reservoir(rows, count, seed, exclusions):
    """Uniform deterministic bounded-memory sample, then randomize nesting order."""
    rng = random.Random(seed)
    selected, eligible = [], 0
    for row in rows:
        if row[0] in exclusions:
            continue
        eligible += 1
        if len(selected) < count:
            selected.append(tuple(row))
        else:
            index = rng.randrange(eligible)
            if index < count:
                selected[index] = tuple(row)
    if len(selected) != count:
        raise ValueError(f"Only {eligible} eligible records, need {count}")
    rng.shuffle(selected)
    return selected, eligible


def prior_reference_paths(roots):
    paths = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths.extend(root.rglob("references.tsv"))
        paths.extend(root.rglob("test_source1.tsv"))
    return sorted(set(paths))


def as_dict(row):
    return dict(zip(FIELDS, row))


def main(args):
    began = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source_paths = prior_reference_paths(args.exclude_roots)
    excluded, sources = set(), {}
    for path in source_paths:
        ids = [r["entity_id"] for r in read_tsv(path)]
        excluded.update(ids)
        sources[str(path.resolve())] = {"references": len(ids), "sha256": sha256(path)}
    fixed = [tuple(r[f] for f in FIELDS) for r in read_tsv(args.existing_train)]
    fixed_ids = [r[0] for r in fixed]
    if len(fixed) != 50000 or len(set(fixed_ids)) != len(fixed):
        raise ValueError("Need exactly 50,000 unique frozen training references")
    trained_ids = [r["entity_id"] for r in read_tsv(args.trained_ids)]
    if fixed_ids != trained_ids:
        raise ValueError("Fixed train order/IDs differ from current model's training IDs")
    excluded.update(fixed_ids)
    tune_rows = list(read_tsv(args.tune))
    dev_rows = list(read_tsv(args.dev))
    tune_ids, dev_ids = {r["entity_id"] for r in tune_rows}, {r["entity_id"] for r in dev_rows}
    if len(tune_ids) != 3000 or len(dev_ids) != 3000:
        raise ValueError("Expected current complete 3000-reference tune and development sets")
    if tune_ids & dev_ids or set(fixed_ids) & (tune_ids | dev_ids):
        raise ValueError("Existing train/tune/development overlap")
    excluded.update(tune_ids | dev_ids)
    db = connect(args.store)
    db.execute("PRAGMA cache_size=-8192")
    for ids, expected in ((fixed_ids, "train"), (tune_ids, "tune"), (dev_ids, "eval")):
        for begin in range(0, len(ids), 1000):
            group = list(ids)[begin:begin + 1000] if isinstance(ids, set) else ids[begin:begin + 1000]
            found = lookup(db, "anchors", "entity_id", group)
            if len(found) != len(group) or any(r["fold"] != expected for r in found.values()):
                raise ValueError(f"Reference missing or inconsistent fold: {expected}")
    extras, train_eligible = reservoir(db.execute(
        "SELECT entity_id,name,address,country FROM anchors WHERE fold='train' ORDER BY entity_id"),
        150000, args.seed, excluded)
    # Audit sampling reads only reference records and fold metadata, never labels.
    audit, audit_eligible = reservoir(db.execute(
        "SELECT entity_id,name,address,country FROM anchors WHERE fold='eval' ORDER BY entity_id"),
        10000, args.seed + 1, excluded)
    db.close()
    extra_ids, audit_ids = {r[0] for r in extras}, {r[0] for r in audit}
    if extra_ids & excluded or audit_ids & excluded or extra_ids & audit_ids:
        raise ValueError("Sampling overlap")
    manifest = {
        "complete": False, "seed": args.seed,
        "source_store": str(Path(args.store).resolve()),
        "source_store_config_sha256": sha256(Path(args.store) / "config.json"),
        "historic_reference_sources": sources, "historic_unique_exclusions": len(excluded),
        "fixed_train_source": str(Path(args.existing_train).resolve()),
        "fixed_train_sha256": sha256(args.existing_train),
        "fixed_training_ids_source": str(Path(args.trained_ids).resolve()),
        "fixed_training_ids_sha256": sha256(args.trained_ids),
        "eligible_train_references": train_eligible, "eligible_eval_references": audit_eligible,
        "selection_policy": "Uniform reservoir by fold over stable entity-ID order; all prior selections excluded from new samples",
        "nesting": "current50k is exact ordered prefix of100k;100k exact ordered prefix of200k",
        "audit_policy": "Fresh10000 eval-fold reference records only; labels neither read nor exported; do not score until all choices frozen",
        "disjointness": "New training,audit and reused tune/dev IDs pairwise disjoint; larger training sets intentionally nested",
        "external_data_used": False, "ground_truth_read": False, "sets": {},
    }
    for name, rows, fold in (("train_100k", fixed + extras[:50000], "train"),
                             ("train_200k", fixed + extras, "train"), ("audit", audit, "eval")):
        folder = out / name
        folder.mkdir()
        path = folder / "references.tsv"
        write_tsv(path, (as_dict(r) for r in rows), FIELDS)
        manifest["sets"][name] = {"references": len(rows), "fold": fold,
            "countries": dict(Counter(r[3] for r in rows)), "sha256": sha256(path),
            "path": str(path.resolve())}
    for name, source, rows, fold in (("tune", args.tune, tune_rows, "tune"),
                                    ("dev", args.dev, dev_rows, "eval")):
        folder = out / name
        folder.mkdir()
        path = folder / "references.tsv"
        shutil.copyfile(source, path)
        if sha256(source) != sha256(path):
            raise ValueError("Copied reference hash mismatch")
        manifest["sets"][name] = {"references": len(rows), "fold": fold,
            "countries": dict(Counter(r["country"] for r in rows)), "sha256": sha256(path),
            "source": str(Path(source).resolve()), "path": str(path.resolve())}
    manifest.update(complete=True, elapsed_seconds=time.perf_counter() - began,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--store", default="artifacts/train_store_v2")
    p.add_argument("--existing-train", default="artifacts/deadline_selection_v1/train/references.tsv")
    p.add_argument("--trained-ids", default="artifacts/deadline_model_50k_v1/training_reference_ids.tsv")
    p.add_argument("--tune", default="artifacts/local_selection_v2/tune/references.tsv")
    p.add_argument("--dev", default="artifacts/local_selection_v2/dev/references.tsv")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--exclude-roots", nargs="+", default=["artifacts/local_selection_v2", "artifacts/deadline_selection_v1", "artifacts/local_sample_v1"])
    main(p.parse_args())
