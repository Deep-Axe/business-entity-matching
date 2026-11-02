"""Small predeclared CPU experiment grid using one fixed retrieval cache."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ber.data import write_json
from ber.retrieval import BASE_CHANNELS, ASCII_CHANNELS


EXPERIMENTS = [
    ("base_k5", 5, BASE_CHANNELS, 6, 400, False),
    ("base_k10", 10, BASE_CHANNELS, 6, 400, False),
    ("base_k20", 20, BASE_CHANNELS, 6, 400, False),
    ("ascii_k10", 10, BASE_CHANNELS + ASCII_CHANNELS, 6, 400, False),
    ("ascii_k20", 20, BASE_CHANNELS + ASCII_CHANNELS, 6, 400, False),
    ("context_base_k10", 10, BASE_CHANNELS, 6, 400, True),
    ("context_ascii_k20", 20, BASE_CHANNELS + ASCII_CHANNELS, 6, 400, True),
    ("deep_context_ascii_k20", 20, BASE_CHANNELS + ASCII_CHANNELS, 8, 600, True),
]


def sweep(train_features, tune_features, dev_features, output, threads=4, experiments=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    chosen = [e for e in EXPERIMENTS if experiments is None or e[0] in experiments]
    if experiments is not None and set(experiments) - {e[0] for e in chosen}:
        raise ValueError("Unknown experiment requested")
    for name, k, channels, depth, iterations, context in chosen:
        destination = output / name
        if (destination / "metrics.json").exists():
            print(f"Already completed {name}; preserving existing run.", flush=True)
            continue
        if destination.exists():
            raise FileExistsError(f"Incomplete run exists; inspect it before retrying: {destination}")
        command = [sys.executable, "-m", "ber.cli", "fit-shards",
                   "--train-features", train_features, "--tune-features", tune_features,
                   "--dev-features", dev_features, "--output", str(destination),
                   "--top-k", str(k), "--channels", *channels, "--depth", str(depth),
                   "--iterations", str(iterations), "--threads", str(threads)]
        if context:
            command.append("--context-features")
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((destination / "metrics.json").read_text())
        print(json.dumps({"experiment": name, "threshold": result["threshold"],
                          "tune_f05": result["tune"]["matching"]["macro_f05"],
                          "dev_f05": result["dev"]["matching"]["macro_f05"],
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    results = []
    for path in output.glob("*/metrics.json"):
        report = json.loads(path.read_text())
        results.append(dict(name=path.parent.name, threshold=report["threshold"],
            tune_macro_f05=report["tune"]["matching"]["macro_f05"],
            dev_macro_f05=report["dev"]["matching"]["macro_f05"],
            dev_precision=report["dev"]["matching"]["pair_precision"],
            dev_recall=report["dev"]["matching"]["pair_recall"],
            dev_blocking_recall=report["dev"]["retrieval"]["pair_recall"],
            dev_mean_candidates=report["dev"]["retrieval"]["mean_candidates"],
            training_seconds=report["training_seconds"], peak_rss_MiB=report["peak_rss_MiB"]))
    results.sort(key=lambda r: (-r["tune_macro_f05"], r["name"]))
    write_json(output / "summary.json", {"selection_criterion": "tuning macro F0.5, not final audit",
                                        "experiments": results})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-features", "tune-features", "dev-features", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--experiments", nargs="+")
    sweep(**vars(parser.parse_args()))
