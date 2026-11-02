"""Freeze a tuning-selected recipe before evaluating the reserved audit."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ber import context_features, features, disk_index, expansion, shards
from ber.data import write_json
from ber.shards import file_hash


def freeze(selections, audit_references, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("A frozen recipe must not be silently replaced")
    entries, signatures = [], set()
    for location in map(Path, selections):
        report = json.loads(location.read_text())
        if report["final_audit_used"]:
            raise ValueError("Selection has already used the audit")
        for candidate in report["comparisons"]:
            calibration = json.loads((Path(candidate["calibration"]) / "threshold.json").read_text())
            cache = json.loads((Path(calibration["features"]) / "config.json").read_text())
            if cache["gallery_fold"] is not None or not cache["labels_available"]:
                raise ValueError("Expected full-gallery labeled tuning features")
            if not calibration["reference_ids_disjoint_from_model_training"]:
                raise ValueError("Tuning/training reference overlap")
            if file_hash(candidate["model"]) != calibration["model_sha256"]:
                raise ValueError("Calibrated model has changed")
            for key in ("threshold", "top_k", "channels", "tuning_macro_f05"):
                if candidate[key] != calibration[key]:
                    raise ValueError(f"Calibration/selection mismatch: {key}")
            signatures.add((cache["references_sha256"], cache["index_config_sha256"],
                            cache["country_block"], cache["store"]))
            entries.append({**candidate, "selection": str(location.resolve()),
                            "selection_sha256": file_hash(location)})
    if len(signatures) != 1 or not entries:
        raise ValueError("Selections must use identical tuning references and gallery")
    entries.sort(key=lambda r: (-r["tuning_macro_f05"], r["top_k"], len(r["channels"]), r["model"]))
    best = entries[0]
    calibration = json.loads((Path(best["calibration"]) / "threshold.json").read_text())
    cache = json.loads((Path(calibration["features"]) / "config.json").read_text())
    recipe = {key: cache[key] for key in ("country_block", "max_df_ratio", "frequency_mode")}
    recipe.update(top_k=best["top_k"], channels=best["channels"], gallery_fold=None)
    if "learned_combined" in best["channels"]:
        recipe["expansion"] = {key: cache["expansion"][key] for key in
                               ("lexicon", "lexicon_sha256", "expansion_k")}
    frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        criterion="Highest full-gallery tuning macro F0.5; smaller k/channels break ties. No dev/audit score used.",
        selected=best, comparisons=entries, recipe=recipe,
        model_sha256=file_hash(best["model"]), final_audit_used=False,
        audit_references=str(Path(audit_references).resolve()),
        audit_references_sha256=file_hash(audit_references), external_data_used=False,
        code_sha256={m.__name__: file_hash(m.__file__) for m in
                     (context_features, features, disk_index, expansion, shards)})
    output.mkdir(parents=True)
    write_json(output / "frozen.json", frozen)
    print(json.dumps({"selected": best, "recipe": recipe}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", nargs="+", required=True)
    parser.add_argument("--audit-references", required=True)
    parser.add_argument("--output", required=True)
    freeze(**vars(parser.parse_args()))
