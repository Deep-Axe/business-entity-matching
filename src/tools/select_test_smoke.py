"""Reproducible country-balanced, unlabeled test sample for pipeline checks."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ber.data import FIELDS, reservoir_add, write_json, write_tsv
from ber.shards import file_hash
from ber.store import connect, record_row


def select(store, output, per_country=250, seed=2028):
    if per_country < 1:
        raise ValueError("per_country must be positive")
    store, output = Path(store), Path(output)
    if json.loads((store / "config.json").read_text())["split"] != "test":
        raise ValueError("Use the unlabeled test store")
    if output.exists():
        raise FileExistsError(output)
    rng, seen, samples = random.Random(seed), Counter(), {}
    db = connect(store)
    for row in db.execute("SELECT * FROM anchors ORDER BY rowid"):
        country = row["country"]
        seen[country] += 1
        reservoir_add(samples.setdefault(country, []), record_row(row),
                      seen[country], per_country, rng)
    db.close()
    records = sorted((r for group in samples.values() for r in group), key=lambda r: r["entity_id"])
    output.mkdir(parents=True)
    write_tsv(output / "references.tsv", records, FIELDS)
    report = dict(store=str(store.resolve()), seed=seed, labels_available=False,
                  country_population=dict(seen), country_sample={c: len(s) for c, s in samples.items()},
                  references=len(records), references_sha256=file_hash(output / "references.tsv"),
                  purpose="Pipeline/throughput smoke test only; no accuracy claims")
    write_json(output / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-country", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2028)
    select(**vars(parser.parse_args()))
