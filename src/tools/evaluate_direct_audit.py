"""Evaluate a previously frozen threshold on separately reserved audit predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from ber.data import write_json
from ber.metrics import score_arrays, retrieval_metrics
from ber.shards import file_hash


def run(predictions, frozen, output):
    directory = Path(predictions)
    choice = json.loads(Path(frozen).read_text())
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"] or config["partitions"] != 1 or not config["labels_available"]:
        raise ValueError("Require a complete labeled audit, not a partial partition")
    for key in ("threshold", "model_sha256", "gate_terms"):
        if config[key] != choice[key]:
            raise ValueError(f"Audit/frozen mismatch: {key}")
    if config["references_sha256"] != choice["audit_references_sha256"]:
        raise ValueError("Not the reserved fresh audit references")
    pieces, offset = {}, 0
    for path in sorted((directory / "shards").glob("[0-9]*/evaluation.npz")):
        committed = json.loads((path.parent / "done.json").read_text())
        if file_hash(path) != committed["output_sha256"]["evaluation.npz"]:
            raise ValueError("Audit prediction data changed after commit")
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        arrays["query_ids"] += offset
        offset += len(arrays["reference_ids"])
        for key, value in arrays.items():
            pieces.setdefault(key, []).append(value)
    a = {key: np.concatenate(values) for key, values in pieces.items()}
    if len(a["reference_ids"]) != choice["audit_count"] or len(set(a["reference_ids"])) != choice["audit_count"]:
        raise ValueError("Incomplete or duplicated audit prediction coverage")
    score = score_arrays(a["query_ids"], a["y"], a["probabilities"], a["true_counts"], choice["threshold"])
    entity_scores = score.pop("entity_scores")
    rng = np.random.default_rng(2047)
    draws = [float(rng.choice(entity_scores, size=len(entity_scores), replace=True).mean()) for _ in range(2000)]
    report = dict(references=len(entity_scores), frozen_manifest_sha256=file_hash(frozen),
        matching=score, retrieval=retrieval_metrics(a["query_ids"], a["y"], a["true_counts"]),
        conditional_bootstrap_95=np.quantile(draws, [.025, .975]).tolist(),
        countries={str(country): dict(references=int((a["countries"] == country).sum()),
            macro_f05=float(entity_scores[a["countries"] == country].mean())) for country in np.unique(a["countries"])},
        caveat="Fixed-model training-domain audit; not hidden-test/France accuracy or uncertainty over model selection")
    write_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "frozen", "output"):
        parser.add_argument(f"--{name}", required=True)
    run(**vars(parser.parse_args()))
