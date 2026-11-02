"""Diagnose channel/filter latency on a small fixed set; not an accuracy estimate."""

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

from ber.data import read_tsv, write_json
from ber.disk_index import DiskRetriever
from ber.shards import ALL_CHANNELS
from ber.text import Record


def profile(index, references, output, count=8, max_df_ratio=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    records = [Record.from_row(r) for r in itertools.islice(read_tsv(references), count)]
    report = {"queries": len(records), "modes": {}, "not_an_accuracy_estimate": True}
    for name, country, fold in (("country_and_fold", True, "train"),
                                ("country_only", True, None), ("unfiltered", False, None)):
        retriever = DiskRetriever(index, country_block=country, fold=fold, max_df_ratio=max_df_ratio)
        report["modes"][name] = {}
        for channel in ALL_CHANNELS:
            retriever.channels = (channel,)
            elapsed = []
            for record in records:
                started = time.perf_counter()
                retriever.candidates(record, 20)
                elapsed.append(time.perf_counter() - started)
            result = {"mean_seconds": statistics.mean(elapsed),
                      "median_seconds": statistics.median(elapsed), "max_seconds": max(elapsed)}
            report["modes"][name][channel] = result
            print(name, channel, result, flush=True)
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-df-ratio", type=float)
    profile(**vars(parser.parse_args()))
