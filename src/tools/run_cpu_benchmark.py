"""Run the approved bounded CPU measurements inside the temporary EC2 venv."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ber.data import write_json


def run(output, sustained_seconds=600):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    hardware = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
    write_json(output / "hardware.json", dict(lscpu=hardware, python=platform.python_version(),
        kernel=platform.release(), started_utc=datetime.now(timezone.utc).isoformat()))
    common = [sys.executable, "tools/benchmark_frozen_retrieval.py",
        "--index", "artifacts/test_index_v2", "--references", "artifacts/test_smoke_selection_v2/references.tsv",
        "--lexicon", "artifacts/training_lexicon_v1", "--per-country", "200"]
    commands = [
        ("one_worker", [*common, "--workers", "1", "--maximum-rounds", "1", "--output", str(output / "one_worker")]),
        ("two_workers", [*common, "--workers", "2", "--minimum-seconds", str(sustained_seconds),
            "--maximum-rounds", "10", "--output", str(output / "two_workers")]),
        ("training", [sys.executable, "-m", "ber.cli", "fit-shards",
            "--train-features", "artifacts/expanded_features_v1/train",
            "--tune-features", "artifacts/expanded_features_v1/tune",
            "--output", str(output / "training"), "--top-k", "20", "--depth", "8",
            "--iterations", "600", "--threads", "2", "--context-features"]),
    ]
    reports = []
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    for name, command in commands:
        started = time.perf_counter()
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("x") as log:
            result = subprocess.run(command, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=2400)
        report = dict(stage=name, seconds=time.perf_counter() - started, exit_code=result.returncode)
        reports.append(report)
        write_json(output / "progress.json", reports)
        print(json.dumps(report), flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; preserve log and stop workload")
    write_json(output / "complete.json", dict(complete=True, stages=reports,
        completed_utc=datetime.now(timezone.utc).isoformat(), full_test_inference_started=False,
        submission_model_replaced=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sustained-seconds", type=int, default=600)
    run(**vars(parser.parse_args()))
