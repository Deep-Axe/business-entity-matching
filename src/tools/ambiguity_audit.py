"""Measure observable ambiguity using supplied labels; no model changes."""

import argparse
import time
from pathlib import Path

from ber.data import write_json
from ber.store import connect


def analyze(store, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    db = connect(store)
    started = time.perf_counter()
    shared_name = db.execute("""
        WITH collisions AS MATERIALIZED (
            SELECT country,name,COUNT(*) AS n FROM anchors GROUP BY country,name HAVING COUNT(*)>1)
        SELECT COUNT(*) AS positive_pairs, COUNT(DISTINCT l.reference_id) AS affected_reference_entities
        FROM records r JOIN labels l ON l.target_id=r.entity_id
        JOIN anchors a ON a.entity_id=l.reference_id
        JOIN collisions c ON c.country=a.country AND c.name=a.name
        WHERE r.address='' AND r.name=a.name
    """).fetchone()
    print(f"Shared exact-name/missing-address check: {dict(shared_name)}", flush=True)
    duplicates = db.execute("""
        SELECT COUNT(*) AS ambiguous_identical_text_groups,
               SUM(n) AS labeled_target_records, MAX(owners) AS largest_distinct_owner_count
        FROM (SELECT r.source,r.country,r.name,COUNT(*) AS n, COUNT(DISTINCT l.reference_id) AS owners
              FROM records r JOIN labels l ON l.target_id=r.entity_id
              WHERE r.address='' GROUP BY r.source,r.country,r.name HAVING COUNT(DISTINCT l.reference_id)>1)
    """).fetchone()
    report = {
        "missing_address_positive_with_shared_exact_reference_name": dict(shared_name),
        "identical_same_source_target_text_with_different_gold_owners": dict(duplicates),
        "seconds": time.perf_counter() - started,
        "interpretation": [
            "Counts use raw, exact strings and empty addresses; they are not a complete ambiguity audit.",
            "Identical source/name/address/country records have no direct textual discriminator between their IDs.",
            "This measures ambiguity, not a computed upper bound on challenge macro F0.5.",
            "No external lookup or generated labels are used.",
        ],
    }
    db.close()
    write_json(output, report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    analyze(**vars(parser.parse_args()))
