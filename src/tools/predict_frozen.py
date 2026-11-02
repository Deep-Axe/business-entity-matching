"""Apply the frozen baseline with provenance checks; never recalibrate."""

import argparse
import importlib
import json
from pathlib import Path

from ber.scalable_model import predict_shards
from ber.shards import file_hash


def run(frozen, features, output, model=None, audit=False, threads=4, resume=False):
    frozen, features = Path(frozen), Path(features)
    choice = json.loads(frozen.read_text())
    recipe = choice["recipe"]
    model = Path(model or choice["selected"]["model"])
    if file_hash(model) != choice["model_sha256"]:
        raise ValueError("Model differs from the frozen selection")
    for module, digest in choice["code_sha256"].items():
        if file_hash(importlib.import_module(module).__file__) != digest:
            raise ValueError(f"Frozen feature/retrieval code changed: {module}")
    cache = json.loads((features / "config.json").read_text())
    for key in ("country_block", "max_df_ratio", "frequency_mode", "gallery_fold"):
        if cache[key] != recipe[key]:
            raise ValueError(f"Frozen/cache recipe mismatch: {key}")
    if cache["feature_code_sha256"] != choice["code_sha256"]["ber.features"]:
        raise ValueError("Cache uses different feature code")
    if "expansion" in recipe:
        for key in ("lexicon_sha256", "expansion_k"):
            if cache.get("expansion", {}).get(key) != recipe["expansion"][key]:
                raise ValueError(f"Frozen expansion mismatch: {key}")
    if audit:
        if not cache["labels_available"] or cache["references_sha256"] != choice["audit_references_sha256"]:
            raise ValueError("Audit requires exactly the reserved labeled references")
    elif cache["labels_available"]:
        raise ValueError("Use --audit for reserved evaluation; otherwise expected unlabeled test")
    result = predict_shards(features, model, choice["selected"]["threshold"], output,
        top_k=recipe["top_k"], channels=recipe["channels"], threads=threads, resume=resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frozen", "features", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model", help="Relocated model, checked against the frozen SHA-256")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    run(**vars(parser.parse_args()))
