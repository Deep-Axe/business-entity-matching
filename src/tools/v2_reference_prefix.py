"""Publish a requested reference-only prefix in a fresh exclusive directory."""
import argparse
from collections import Counter
import itertools
from pathlib import Path

from ber.data import FIELDS, read_tsv, write_json, write_tsv
from ber.shards import file_hash


def run(source, output, count):
    if count <= 0:
        raise ValueError("Positive prefix size required")
    source, output = Path(source).resolve(), Path(output).absolute()
    before = file_hash(source)
    rows = list(itertools.islice(read_tsv(source), count))
    if len(rows) != count or len({r["entity_id"] for r in rows}) != count:
        raise ValueError("Source too short or duplicate prefix reference IDs")
    if any(set(r) != set(FIELDS) or not r["entity_id"].startswith("S1-") for r in rows):
        raise ValueError("Unexpected reference schema or IDs")
    if file_hash(source) != before:
        raise ValueError("Source changed during prefix extraction")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "references.tsv"
    write_tsv(destination, rows, FIELDS)
    if list(read_tsv(destination)) != rows:
        raise ValueError("Serialized prefix differs")
    report = dict(complete=True, references=count, source=str(source), source_sha256=before,
        references_sha256=file_hash(destination), countries=dict(Counter(r["country"] for r in rows)),
        labels_accessed=False, source_modified=False)
    write_json(output / "manifest.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    run(**vars(parser.parse_args()))
