"""Paired development-set uncertainty, without choosing on final-audit results."""

import argparse
import json
from pathlib import Path

import numpy as np

from ber.data import read_tsv, write_json


def compare(sweep, output, reference="base_k20", resamples=2000, seed=2031):
    sweep, output = Path(sweep), Path(output)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((sweep / "summary.json").read_text())
    baseline = list(read_tsv(sweep / reference / "dev/entity_metrics.tsv"))
    ids = [r["source1_entity_id"] for r in baseline]
    base_scores = np.array([float(r["macro_f05"]) for r in baseline])
    rng = np.random.default_rng(seed)
    result = []
    for experiment in summary["experiments"]:
        rows = list(read_tsv(sweep / experiment["name"] / "dev/entity_metrics.tsv"))
        if [r["source1_entity_id"] for r in rows] != ids:
            raise ValueError("Development entity alignment differs across experiments")
        scores = np.array([float(r["macro_f05"]) for r in rows])
        difference = scores - base_scores
        draws = [float(difference[rng.integers(0, len(ids), len(ids))].mean()) for _ in range(resamples)]
        result.append({**experiment, "dev_delta_vs_reference": float(difference.mean()),
                       "paired_bootstrap_95_percent_interval": list(map(float, np.quantile(draws, [.025, .975])))})
    report = dict(reference=reference, selected_by_tuning=summary["experiments"][0]["name"],
        entities=len(ids), experiments=result,
        caveat="Conditional paired entity bootstrap. It does not account for model selection, country shift, label noise, or a different gallery. Final audit is unused.")
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default="base_k20")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2031)
    compare(**vars(parser.parse_args()))
