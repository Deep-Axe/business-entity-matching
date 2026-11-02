"""Cheap read-only corpus checks on the completed training record store."""

import argparse
import json
import time
from pathlib import Path

from ber.data import write_json
from ber.store import connect


def analyze(store, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    db = connect(store)
    queries = {
        "reference_raw_name_collisions": """SELECT COUNT(*) AS colliding_country_name_groups,
            SUM(n) AS references_in_groups, MAX(n) AS largest_group FROM
            (SELECT country,name,COUNT(*) AS n FROM anchors GROUP BY country,name HAVING COUNT(*)>1)""",
        "reference_raw_address_collisions": """SELECT COUNT(*) AS colliding_country_address_groups,
            SUM(n) AS references_in_groups, MAX(n) AS largest_group FROM
            (SELECT country,address,COUNT(*) AS n FROM anchors GROUP BY country,address HAVING COUNT(*)>1)""",
        "target_missing_addresses_by_source_country": """SELECT source,country,COUNT(*) AS targets,
            SUM(address='') AS missing_addresses FROM records GROUP BY source,country""",
        "match_count_distribution": "SELECT n_matches,COUNT(*) AS references_count FROM gold_rows GROUP BY n_matches",
    }
    report = {"store": str(Path(store).resolve()), "checks": {}}
    for name, sql in queries.items():
        report["checks"][name] = [dict(row) for row in db.execute(sql)]
        print(f"completed {name} after {time.perf_counter() - started:.1f}s", flush=True)
    report["seconds"] = time.perf_counter() - started
    report["notes"] = ["Collisions are exact raw-text comparisons within country, not normalized identities.",
                       "An address or a business name alone may refer to multiple reference entities."]
    db.close()
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    analyze(**vars(parser.parse_args()))
