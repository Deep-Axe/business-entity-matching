"""Recompute one committed cloud test shard locally and compare all pair features."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

from aws_baseline_transfer import ssh
from aws_baseline_jobs import REMOTE
from ber.data import read_tsv, write_json
from ber.deadline_retrieval import retrievers
from ber.direct_features import make_arrays
from ber.disk_index import TargetStore
from ber.shards import file_hash


def run(run, output):
    run, output = Path(run).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = json.loads((run / "state.json").read_text())
    host = state["hosts"][0]
    destination = output / "cloud_shard"
    destination.mkdir()
    transport = shlex.join(ssh(run, host)[:-1])
    source = REMOTE+"/artifacts/deadline_test_features_cloud_0_v1/shards/000000/"
    subprocess.run(["rsync", "-a", "-e", transport, "ubuntu@"+host["ip"]+":"+source, str(destination)+"/"], check=True)
    committed = json.loads((destination / "done.json").read_text())
    if file_hash(destination / "features.npz") != committed["output_sha256"]["features.npz"]:
        raise ValueError("Cloud shard checksum mismatch")
    base, extra = retrievers("artifacts/test_index_v2", "artifacts/training_lexicon_v1", 1)
    targets = TargetStore("artifacts/test_store_v2")
    rows = list(read_tsv(destination / "references.tsv"))
    local, timing = make_arrays(rows, base, extra, targets)
    with np.load(destination / "features.npz", allow_pickle=False) as z:
        cloud = {key: z[key] for key in z.files}
    positions = {(int(q), str(t)): i for i, (q, t) in enumerate(zip(local["query_ids"], local["target_ids"]))}
    keys = [(int(q), str(t)) for q, t in zip(cloud["query_ids"], cloud["target_ids"])]
    if set(keys) != set(positions):
        raise ValueError("Cloud/local candidate IDs differ")
    order = [positions[key] for key in keys]
    for name in ("X", "channel_scores", "channel_ranks", "y"):
        np.testing.assert_array_equal(cloud[name], local[name][order])
    for name in ("reference_ids", "countries", "true_counts", "channel_order"):
        np.testing.assert_array_equal(cloud[name], local[name])
    targets.close()
    report = dict(passed=True, exact_cross_host_features=True, references=len(rows),
                  candidate_pairs=len(keys), cloud_features_sha256=file_hash(destination / "features.npz"), timing=timing)
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))
