"""Streaming format/ID validation against the immutable record store."""

import itertools
import json
import time
from collections import OrderedDict
from pathlib import Path

from .data import read_tsv, write_json
from .store import batches, connect, lookup


def _ids(value):
    if not value:
        return []
    result = value.split(",")
    if any(not x or x != x.strip() for x in result):
        raise ValueError("Blank or whitespace-padded target ID")
    if len(result) != len(set(result)):
        raise ValueError("Duplicate target ID within a reference row")
    return result


def validate_exports(store, references, matching, candidates, output_report=None):
    db = connect(store)
    started = time.perf_counter()
    seen_refs, valid_cache = set(), OrderedDict()
    counts = {"reference_rows": 0, "candidate_pairs": 0, "predicted_pairs": 0,
              "empty_predictions": 0}
    iterator = itertools.zip_longest(read_tsv(references), read_tsv(matching), read_tsv(candidates))
    try:
        for group in batches(iterator, 200):
            wanted = set()
            for reference, prediction, candidate in group:
                if reference is None or prediction is None or candidate is None:
                    raise ValueError("Reference/matching/candidate row counts differ")
                eid = reference["entity_id"]
                if eid in seen_refs:
                    raise ValueError("Duplicate reference ID")
                seen_refs.add(eid)
                if prediction["source1_entity_id"] != eid or candidate["source1_entity_id"] != eid:
                    raise ValueError("Reference order or IDs differ across exports")
                matched, possible = _ids(prediction["matched_entity_ids"]), _ids(candidate["candidate_entity_ids"])
                if not set(matched) <= set(possible):
                    raise ValueError("Predicted match absent from actual matcher inputs")
                wanted.update(possible)
                counts["reference_rows"] += 1
                counts["candidate_pairs"] += len(possible)
                counts["predicted_pairs"] += len(matched)
                counts["empty_predictions"] += not matched
            unknown = list(wanted - valid_cache.keys())
            found = lookup(db, "records", "entity_id", unknown)
            if len(found) != len(unknown):
                raise ValueError(f"Unknown target IDs: {sorted(set(unknown) - found.keys())[:5]}")
            for eid in wanted:
                valid_cache.pop(eid, None)
                valid_cache[eid] = True
            while len(valid_cache) > 50000:
                valid_cache.popitem(last=False)
        report = {"valid": True, **counts,
                  "seconds": round(time.perf_counter() - started, 3),
                  "scope": "Exact supplied reference list; target membership checked against SQLite store"}
        if output_report:
            write_json(output_report, report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        db.close()
