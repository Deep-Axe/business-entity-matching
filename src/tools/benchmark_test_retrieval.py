"""Small country-stratified throughput benchmark; test labels are never inferred."""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ber.data import read_tsv, write_json
from ber.disk_index import DiskRetriever
from ber.text import Record


def run(index, references, output, per_country=30, top_k=20, channels=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    groups = defaultdict(list)
    for row in read_tsv(references):
        groups[row["country"]].append(row)
    rng = random.Random(2030)
    selected = []
    for country, rows in sorted(groups.items()):
        selected.extend(rng.sample(rows, min(per_country, len(rows))))
    rng.shuffle(selected)
    retriever = DiskRetriever(index, transliteration=True, channels=channels,
                               max_df_ratio=.01, frequency_mode="gate")
    times, candidates = defaultdict(list), defaultdict(list)
    for row in selected:
        record = Record.from_row(row)
        stamp = time.perf_counter()
        hits = retriever.candidates(record, top_k)
        times[record.country].append(time.perf_counter() - stamp)
        candidates[record.country].append(len(hits))
    result = dict(top_k=top_k, channels=list(retriever.channels), max_df_ratio=.01,
        frequency_mode="gate", labels_available=False,
        caveat="Small sample, warm shared filesystem cache, potentially concurrent CPU work. Not an AWS or full-job runtime promise.",
        countries={c: dict(queries=len(values), mean_query_seconds=float(np.mean(values)),
            p50_query_seconds=float(np.median(values)), p95_query_seconds=float(np.quantile(values, .95)),
            mean_candidates=float(np.mean(candidates[c]))) for c, values in times.items()})
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "references", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--per-country", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--channels", nargs="+")
    run(**vars(parser.parse_args()))
