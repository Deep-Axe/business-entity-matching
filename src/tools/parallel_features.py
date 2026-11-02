"""Run independent feature shards in separate processes, then verify completeness."""

import argparse
import json
import os
import subprocess
import sys
import time

from ber.shards import featurize_shards, finalize_features
from ber.data import write_json
from pathlib import Path


def run(args):
    values = vars(args).copy()
    workers = values.pop("workers")
    if workers < 1:
        raise ValueError("workers must be positive")
    started = time.perf_counter()
    featurize_shards(**values, initialize_only=True)
    command = [sys.executable, "-m", "ber.cli", "features"]
    for key, value in values.items():
        if key == "country_block":
            if not value:
                command.append("--no-country-block")
        elif key == "resume":
            continue  # Initialization has created/verified the shared output.
        elif isinstance(value, bool):
            if value:
                command.append("--" + key.replace("_", "-"))
        elif value is not None:
            command.extend(("--" + key.replace("_", "-"), str(value)))
    command.extend(("--resume", "--workers", str(workers)))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    processes = []
    try:
        for worker in range(workers):
            processes.append(subprocess.Popen([*command, "--worker-id", str(worker)], env=environment))
        while any(p.poll() is None for p in processes):
            failures = [p.returncode for p in processes if p.returncode not in (None, 0)]
            if failures:
                raise RuntimeError(f"Feature worker failed: {failures}")
            time.sleep(1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Feature worker failed")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    report = finalize_features(args.output)
    timing = {"references": report["references"], "candidate_pairs": report["candidate_pairs"],
              "workers": workers, "wall_seconds": time.perf_counter() - started,
              "resumed": args.resume,
              "note": "Wall time for this invocation, including only unfinished shards on resume."}
    write_json(Path(args.output) / f"execution.{time.time_ns()}.json", timing)
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "index", "references", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gallery-fold", choices=("train", "tune", "eval", "test"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--transliteration", action="store_true")
    parser.add_argument("--query-batch-size", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=10)
    parser.add_argument("--max-df-ratio", type=float)
    parser.add_argument("--frequency-mode", choices=("drop", "gate"), default="drop")
    parser.add_argument("--lexicon", help="training lexicon directory containing lexicon.json")
    parser.add_argument("--gate-terms", type=int, default=0)
    parser.add_argument("--no-country-block", dest="country_block", action="store_false")
    run(parser.parse_args())
