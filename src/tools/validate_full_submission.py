"""Strict streaming checks of both complete outputs; keep only actual target IDs.

Avoid the official validator's corpus-sized candidate mapping. The original
official CLI is additionally run on full matching_results.tsv with --check-ids.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from ber.data import read_tsv, write_json
from ber.shards import file_hash


def target_ids(test_dir):
    ids = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        with (Path(test_dir) / name).open(encoding="utf-8") as handle:
            header = next(handle).rstrip("\r\n").split("\t")
            if header[0] != "entity_id":
                raise ValueError("Unknown target schema")
            for line in handle:
                value = line.partition("\t")[0]
                if value in ids or not value.startswith(("S2-", "S3-")):
                    raise ValueError("Invalid original target IDs")
                ids.add(value)
    return ids


def parse(row, column, targets):
    values = row[column].split(",") if row[column] else []
    if len(values) != len(set(values)):
        raise ValueError("Duplicate ID within output list")
    for value in values:
        if not value.startswith(("S2-", "S3-")) or value not in targets:
            raise ValueError("Invalid target ID in output")
    return set(values)


def run(test_dir, output, report):
    started = time.perf_counter()
    output, test_dir = Path(output), Path(test_dir)
    targets = target_ids(test_dir)
    print(f"Loaded {len(targets):,} original target IDs", flush=True)
    seen, counts = set(), dict(references=0, candidate_pairs=0, predicted_pairs=0, empty_predictions=0)
    countries = {}
    streams = (read_tsv(test_dir / "test_source1.tsv"), read_tsv(output / "matching_results.tsv"), read_tsv(output / "candidate_pairs.tsv"))
    for reference, matching, candidate in itertools.zip_longest(*streams):
        if any(value is None for value in (reference, matching, candidate)):
            raise ValueError("Output/reference row counts differ")
        if set(matching) != {"source1_entity_id", "matched_entity_ids"} or set(candidate) != {"source1_entity_id", "candidate_entity_ids"}:
            raise ValueError("Incorrect output columns")
        eid = reference["entity_id"]
        if eid in seen or matching["source1_entity_id"] != eid or candidate["source1_entity_id"] != eid:
            raise ValueError("Duplicate/missing/out-of-order reference")
        seen.add(eid)
        matches, candidates = parse(matching, "matched_entity_ids", targets), parse(candidate, "candidate_entity_ids", targets)
        if not matches <= candidates:
            raise ValueError("Final match was not considered by matcher")
        counts["references"] += 1
        counts["candidate_pairs"] += len(candidates)
        counts["predicted_pairs"] += len(matches)
        counts["empty_predictions"] += not matches
        countries[reference["country"]] = countries.get(reference["country"], 0)+1
        if counts["references"] % 100000 == 0:
            print(json.dumps(dict(**counts, elapsed_seconds=time.perf_counter()-started)), flush=True)
    result = dict(valid=True, **counts, countries=countries, target_records=len(targets),
        seconds=time.perf_counter()-started, checks="Complete ordered S1 coverage, exact columns, duplicates, every S2/S3 ID, matches subset of candidates",
        sha256={name: file_hash(output / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")},
        note="Format/ID validation only; no hidden test accuracy is known")
    write_json(report, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test-dir", "output", "report"):
        parser.add_argument(f"--{name}", required=True)
    run(**vars(parser.parse_args()))
