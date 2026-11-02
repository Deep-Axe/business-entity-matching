"""Repeatable full-index, seven-channel CPU benchmark; no accuracy claims."""

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import random
import resource
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ber.data import read_tsv, write_json
from ber.disk_index import DiskRetriever
from ber.expansion import ExpansionRetriever
from ber.shards import file_hash
from ber.text import Record


_base = _extra = None


def initialize(index, lexicon):
    global _base, _extra
    _base = DiskRetriever(index, transliteration=True, max_df_ratio=.01, frequency_mode="gate")
    _extra = ExpansionRetriever(index, lexicon, max_df_ratio=.01, frequency_mode="gate")


def search(row):
    record = Record.from_row(row)
    started, cpu = time.perf_counter(), time.process_time()
    found = {row_id for row_id, _ in _base.candidates(record, 20)}
    found.update(row_id for row_id, _ in _extra.candidates(record, 20))
    seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
    signature = hashlib.sha256(
        (record.entity_id + ":" + ",".join(map(str, sorted(found)))).encode()).hexdigest()
    return dict(reference_id=record.entity_id, country=record.country, seconds=seconds,
                cpu_seconds=cpu_seconds, candidates=len(found), candidate_ids_sha256=signature,
                worker_peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def run(index, references, lexicon, output, workers=2, per_country=200,
        minimum_seconds=0, maximum_rounds=10, seed=2035):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if min(workers, per_country, maximum_rounds) < 1 or minimum_seconds < 0:
        raise ValueError("Invalid benchmark limits")
    groups = defaultdict(list)
    for row in read_tsv(references):
        groups[row["country"]].append(row)
    rng = random.Random(seed)
    selected = [r for country in sorted(groups)
                for r in rng.sample(groups[country], min(per_country, len(groups[country])))]
    rng.shuffle(selected)
    if not selected:
        raise ValueError("No benchmark references")
    output.mkdir(parents=True)
    manifest = json.loads((Path(index) / "manifest.json").read_text())
    metadata = dict(index_records=manifest["indexed_records"], references_sha256=file_hash(references),
        lexicon_sha256=file_hash(Path(lexicon) / "lexicon.json"),
        benchmark_code_sha256=file_hash(__file__), workers=workers, seed=seed,
        references_per_round=len(selected), per_country=per_country, minimum_seconds=minimum_seconds,
        maximum_rounds=maximum_rounds, machine=platform.machine(), cpu_count=os.cpu_count(),
        python=platform.python_version(), kernel=platform.release(), top_k=20,
        channels="six base plus frozen learned_combined", labels_available=False,
        scope="Search only: excludes SQLite lookup, pair features and model scoring.",
        caveat="Repeated fixed queries and warm filesystem caches; short CPU bursts do not prove multi-day Flex throughput.")
    write_json(output / "config.json", metadata)
    started, rounds = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
            initargs=(index, lexicon), mp_context=multiprocessing.get_context("spawn")) as executor:
        # Small warm-up makes process startup explicit and leaves all scored queries unchanged.
        list(executor.map(search, selected[:workers * 2], chunksize=1))
        warmup_seconds = time.perf_counter() - started
        for round_number in range(maximum_rounds):
            stamp = time.perf_counter()
            records = list(executor.map(search, selected, chunksize=5))
            elapsed = time.perf_counter() - stamp
            checksum = hashlib.sha256("".join(r["candidate_ids_sha256"] for r in records).encode()).hexdigest()
            report = dict(round=round_number, wall_seconds=elapsed,
                references=len(records), references_per_second=len(records) / elapsed,
                sum_worker_cpu_seconds=sum(r["cpu_seconds"] for r in records),
                maximum_worker_peak_rss_MiB=max(r["worker_peak_rss_MiB"] for r in records),
                candidate_pairs=sum(r["candidates"] for r in records), candidate_ids_sha256=checksum,
                countries={country: dict(queries=sum(r["country"] == country for r in records),
                    mean_query_seconds=float(np.mean([r["seconds"] for r in records if r["country"] == country])),
                    p95_query_seconds=float(np.quantile([r["seconds"] for r in records if r["country"] == country], .95)))
                    for country in sorted(groups)})
            write_json(output / f"round_{round_number:02d}.json", report)
            rounds.append(report)
            print(json.dumps(report), flush=True)
            if sum(r["wall_seconds"] for r in rounds) >= minimum_seconds:
                break
    write_json(output / "summary.json", dict(config=metadata, warmup_seconds=warmup_seconds,
        total_wall_seconds=time.perf_counter() - started, rounds=rounds,
        deterministic_candidates_across_rounds=len({r["candidate_ids_sha256"] for r in rounds}) == 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "references", "lexicon", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--per-country", type=int, default=200)
    parser.add_argument("--minimum-seconds", type=float, default=0)
    parser.add_argument("--maximum-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2035)
    run(**vars(parser.parse_args()))
