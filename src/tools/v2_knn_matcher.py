"""Bounded, tune-selected transfer diagnostic for full-gallery kNN additions.

Preserve original 38 lexical pair features. New pairs get truthful empty lexical
hits, not invented BM25 evidence. Recompute candidate-neighborhood context on
each complete union. No training, audit, external data, or submission mutation.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from catboost import CatBoostClassifier

from ber.context_features import context_matrix
from ber.data import read_tsv
from ber.features import FEATURE_NAMES, pair_features
from ber.store import connect, lookup, record_row
from ber.text import Record
from v2_cascade_cache import validate_inputs
from v2_cascade_train import entity_scores
from v2_cpu_features import View, Weighting, extra_features, EXTRA_NAMES
from v2_run_pipeline import NAMES, code_hashes
from v2_sparse_fullgallery import sha, read_json, new_json


BASE_THRESHOLD = .60
CHANNELS = ("word", "char", "union")
KS = (10, 20, 50)
ADDITIVE_THRESHOLDS = (.60, .75, .90, .97)
UNION_THRESHOLDS = (.55, .60, .65, .70)


def verify_collected(gallery, scored, queries):
    """Verify compact downloaded commit chains without requiring large count blocks."""
    gallery, scored, queries = map(Path, (gallery, scored, queries))
    gc, gm = read_json(gallery / "config.json"), read_json(gallery / "manifest.json")
    sc, sm = read_json(scored / "config.json"), read_json(scored / "manifest.json")
    qm = read_json(queries / "manifest.json")
    if not (gm.get("complete") and gm.get("full_gallery") and sm.get("complete") and sm.get("full_gallery")):
        raise ValueError("Complete full-gallery retrieval is mandatory")
    if gc.get("labels_read", True) or sc.get("labels_read", True) or sm.get("labels_read", True):
        raise ValueError("Ranking must be explicitly label-free")
    if not (gm["rows"] == gc["requested_rows"] == gc["total_gallery_rows"] == sm["gallery_rows"]):
        raise ValueError("Gallery coverage identity mismatch")
    checks = [(sha(gallery / "config.json"), gm["config_sha256"]),
        (sha(gallery / "idf.npy"), gm["idf_sha256"]), (sha(gallery / "df.npy"), gm["df_sha256"]),
        (sha(gallery / "manifest.json"), sc["gallery_manifest_sha256"]),
        (sha(scored / "config.json"), sm["config_sha256"]),
        (sha(scored / "neighbors.tsv"), sm["neighbors_sha256"]),
        (sha(queries / "manifest.json"), sc["queries_manifest_sha256"]),
        (sha(queries / "queries.tsv"), qm["queries_sha256"]),
        (qm["queries_sha256"], sc["queries_sha256"])]
    if any(a != b for a, b in checks):
        raise ValueError("Collected provenance checksum mismatch")
    if sc["channel"] != gc["representation"]["channel"] or sc["k"] < 50 or sm["queries"] != qm["queries"]:
        raise ValueError("Channel/k/query identity mismatch")
    nblocks = math.ceil(gm["rows"] / gc["block_size"])
    if [b["block"] for b in gm["blocks"]] != list(range(nblocks)) or [b["block"] for b in sm["blocks"]] != list(range(nblocks)):
        raise ValueError("Exact ordered block coverage is missing")
    for source, score in zip(gm["blocks"], sm["blocks"]):
        block = source["block"]
        gp, sp = gallery / "blocks" / f"{block:06d}" / "done.json", scored / "blocks" / f"{block:06d}" / "done.json"
        if sha(gp) != source["done_sha256"] or sha(sp) != score["done_sha256"]:
            raise ValueError("Collected block commit checksum mismatch")
        gd, sd = read_json(gp), read_json(sp)
        begin, end = block * gc["block_size"] + 1, min((block + 1) * gc["block_size"], gm["rows"])
        if (gd["config_sha256"] != gm["config_sha256"] or gd["first_row_id"] != begin or gd["last_row_id"] != end
                or gd["rows"] != end-begin+1 or sd["source_done_sha256"] != source["done_sha256"]
                or sd["config_sha256"] != sm["config_sha256"]):
            raise ValueError("Collected commit coverage/config mismatch")
    return {"channel": sc["channel"], "gallery_manifest_sha256": sha(gallery / "manifest.json"),
            "score_manifest_sha256": sha(scored / "manifest.json"), "source": gc["source"],
            "neighbors_sha256": sm["neighbors_sha256"], "queries_sha256": qm["queries_sha256"],
            "rows": gm["rows"], "blocks": nblocks, "runner_sha256": gc["runner_sha256"]}


def neighbors_for(scored, rows):
    expected = {r["entity_id"]: r for r in rows}
    neighbors = {r: [] for r in expected}
    for row in read_tsv(Path(scored) / "neighbors.tsv"):
        rid = row["source1_entity_id"]
        if rid not in expected or row["country"] != expected[rid]["country"] or row["cohort"] != expected[rid]["cohort"]:
            raise ValueError("Neighbor query metadata mismatch")
        if int(row["rank"]) != len(neighbors[rid])+1 or row["target_id"] in neighbors[rid]:
            raise ValueError("Invalid or duplicate ranked neighbor")
        neighbors[rid].append(row["target_id"])
    return neighbors


def load_baseline(features, probabilities, matrix, rows, store, index, model_sha):
    config, manifest, binding, _, _, p = validate_inputs(features, probabilities, matrix, store, index)
    if config["gallery_fold"] is not None or config["gate_terms"] != 0 or binding["model_sha256"] != model_sha:
        raise ValueError("Baseline must be same enhanced model on gate0 full gallery")
    wanted = {r["entity_id"]: r for r in rows}
    result, truth = {}, {}
    for item in manifest["shards"]:
        folder = Path(features) / "shards" / item["shard"]
        with np.load(folder / "features.npz", allow_pickle=False) as z:
            refs = z["reference_ids"]
            if not any(str(r) in wanted for r in refs):
                continue
            x, targets, queries = z["X"], z["target_ids"], z["query_ids"]
            shardp = p[item["pair_start"]:item["pair_end"]]
            for q, rid in enumerate(refs):
                rid = str(rid)
                if rid in wanted:
                    take = queries == q
                    if rid in result:
                        raise ValueError("Duplicate baseline query")
                    result[rid] = {"X": x[take].copy(), "targets": targets[take].copy(), "p": shardp[take].copy()}
        done = read_json(folder / "done.json")
        for name in ("ground_truth.tsv", "references.tsv"):
            if sha(folder / name) != done["output_sha256"][name]:
                raise ValueError("Baseline reference/gold checksum mismatch")
        for row in read_tsv(folder / "references.tsv"):
            if row["entity_id"] in wanted:
                if any(row[k] != wanted[row["entity_id"]][k] for k in row):
                    raise ValueError("Prepared query text differs from baseline")
        for row in read_tsv(folder / "ground_truth.tsv"):
            if row["source1_entity_id"] in wanted:
                truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    if set(result) != set(wanted) or set(truth) != set(wanted):
        raise ValueError("Baseline/gold coverage differs from fixed queries")
    return result, truth, {"binding": binding, "matrix_manifest_sha256": sha(matrix),
                           "source_config_sha256": sha(Path(features) / "config.json")}


def assemble_master(rows, baseline, neighbors, truth, target_rows, weights):
    """Pure ordered union feature construction; supplied gold is metadata only."""
    names, qids, targets, x, extras, oldp, isbase, wordrank, charrank, labels = [], [], [], [], [], [], [], [], [], []
    for q, row in enumerate(rows):
        rid = row["entity_id"]
        names.append(rid)
        reference, left = Record.from_row(row), View.from_row(row)
        old = baseline[rid]
        original = list(map(str, old["targets"]))
        if len(original) != len(set(original)):
            raise ValueError("Duplicate baseline candidate")
        ranked = {c: {eid: i+1 for i, eid in enumerate(neighbors[c][rid])} for c in ("word", "char")}
        additions = sorted(set(ranked["word"]) | set(ranked["char"]))
        ids = original + sorted(set(additions) - set(original))
        for j, eid in enumerate(ids):
            targetrow = record_row(target_rows[eid])
            if targetrow["country"] != row["country"]:
                raise ValueError("Cross-country candidate cannot enter union")
            base = j < len(original)
            qids.append(q); targets.append(eid); isbase.append(base)
            # No ablation helper: zero lexical evidence is deliberate and must survive.
            x.append(old["X"][j] if base else pair_features(reference, Record.from_row(targetrow), {}))
            extras.append(extra_features(left, View.from_row(targetrow), weights))
            oldp.append(float(old["p"][j]) if base else np.nan)
            wordrank.append(ranked["word"].get(eid, 0)); charrank.append(ranked["char"].get(eid, 0))
            labels.append(int(eid in truth[rid]))
    return {"X": np.asarray(x, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)),
            "extras": np.asarray(extras, dtype=np.float32).reshape(-1, len(EXTRA_NAMES)),
            "query_ids": np.asarray(qids, dtype=np.int32), "target_ids": np.asarray(targets),
            "reference_ids": np.asarray(names), "country": np.asarray([r["country"] for r in rows]),
            "true_counts": np.asarray([len(truth[r["entity_id"]]) for r in rows], dtype=np.int32),
            "baseline_p": np.asarray(oldp), "is_baseline": np.asarray(isbase, dtype=bool),
            "word_rank": np.asarray(wordrank, dtype=np.int16), "char_rank": np.asarray(charrank, dtype=np.int16),
            "y": np.asarray(labels, dtype=np.int8)}


def recipe_mask(a, channel, k):
    chosen = a["is_baseline"].copy()
    for c in (("word", "char") if channel == "union" else (channel,)):
        rank = a[c + "_rank"]
        chosen |= (rank > 0) & (rank <= k)
    return chosen


def matrix_for(a, take):
    # Context sees the entire policy union, even if only additions are accepted later.
    original = {"X": a["X"][take], "query_ids": a["query_ids"][take], "reference_ids": a["reference_ids"]}
    return np.column_stack((context_matrix(original), a["extras"][take]))


def base_decisions(a):
    return a["is_baseline"] & (a["baseline_p"] >= BASE_THRESHOLD)


def decisions(a, take, probabilities, mode, threshold):
    if mode == "baseline":
        return base_decisions(a)
    result = np.zeros(len(a["y"]), dtype=bool)
    result[take] = probabilities >= threshold
    if mode == "additive":
        result[a["is_baseline"]] = base_decisions(a)[a["is_baseline"]]
    elif mode != "union":
        raise ValueError("Unknown decision policy")
    return result


def metrics(a, selected):
    values = entity_scores(a, selected.astype(np.float32), .5)
    tp = int((selected & (a["y"] == 1)).sum()); fp = int((selected & (a["y"] == 0)).sum())
    return {"macro_f05": float(values.mean()), "tp": tp, "fp": fp,
            "precision": tp/max(tp+fp, 1), "recall": tp/max(int(a["true_counts"].sum()), 1),
            "by_country": {str(c): {"references": int((a["country"] == c).sum()),
                "macro_f05": float(values[a["country"] == c].mean())} for c in np.unique(a["country"])}}


def best_policy(grid, mode=None):
    eligible = [r for r in grid if mode is None or r["mode"] in (mode, "baseline")]
    return max(eligible, key=lambda r: (r["macro_f05"], r["mode"] == "baseline",
        r["mode"] == "additive", -r["candidate_pairs"], r["threshold"]))


def tune_policies(a, predict):
    baseline = {"mode": "baseline", "channel": "none", "k": 0, "threshold": BASE_THRESHOLD,
                "candidate_pairs": int(a["is_baseline"].sum()), **metrics(a, base_decisions(a))}
    grid = [baseline]
    for channel in CHANNELS:
        for k in KS:
            take = recipe_mask(a, channel, k)
            p = predict(matrix_for(a, take))
            for mode, thresholds in (("additive", ADDITIVE_THRESHOLDS), ("union", UNION_THRESHOLDS)):
                for threshold in thresholds:
                    selected = decisions(a, take, p, mode, threshold)
                    grid.append({"mode": mode, "channel": channel, "k": k, "threshold": threshold,
                                 "candidate_pairs": int(take.sum()), **metrics(a, selected)})
    return {"overall": best_policy(grid), "additive": best_policy(grid, "additive"),
            "union": best_policy(grid, "union")}, grid


def compare(a, selected, base=None):
    base = base_decisions(a) if base is None else base
    delta = entity_scores(a, selected.astype(np.float32), .5) - entity_scores(a, base.astype(np.float32), .5)
    rng = np.random.default_rng(20260925)
    boot = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(3000)])
    added, removed = selected & ~base, base & ~selected
    return {"delta_macro_f05": float(delta.mean()), "paired_owner_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "added_tp": int((added & (a["y"] == 1)).sum()), "added_fp": int((added & (a["y"] == 0)).sum()),
            "removed_tp": int((removed & (a["y"] == 1)).sum()), "removed_fp": int((removed & (a["y"] == 0)).sum()),
            "improved_owners": int((delta > 0).sum()), "worsened_owners": int((delta < 0).sum()),
            "conditional_on_tune_selected_policy": True}


def run(args):
    started = time.perf_counter()
    out, querydir = Path(args.output), Path(args.queries)
    if out.exists():
        raise FileExistsError(out)
    if any("audit" in str(v).lower() for v in vars(args).values()):
        raise ValueError("Reserved audit must not be used")
    retrieval = {}
    for gallery, scores in zip(args.galleries, args.scores):
        verified = verify_collected(gallery, scores, querydir)
        if verified["channel"] in retrieval:
            raise ValueError("Duplicate channel")
        retrieval[verified["channel"]] = {**verified, "score_path": scores}
    if set(retrieval) != {"word", "char"} or retrieval["word"]["source"] != retrieval["char"]["source"]:
        raise ValueError("Both matching full-gallery word and char sources required")
    evaluation = read_json(Path(args.evaluation) / "report.json")
    qm = read_json(querydir / "manifest.json")
    if evaluation["queries_sha256"] != qm["queries_sha256"]:
        raise ValueError("Retrieval evaluation and matcher queries differ")
    for c in ("word", "char"):
        if evaluation["provenance"][c]["manifest_sha256"] != retrieval[c]["score_manifest_sha256"]:
            raise ValueError("Retrieval evaluation score provenance differs")
    rows = list(read_tsv(querydir / "queries.tsv"))
    if len(rows) != 800 or len({r["entity_id"] for r in rows}) != 800 or any(sum(r["cohort"] == c for r in rows) != 400 for c in ("tune", "dev")):
        raise ValueError("This bounded diagnostic requires exactly400 tune +400 dev distinct queries")
    indexconfig = read_json(Path(args.index) / "config.json")
    if (indexconfig["store_config_sha256"] != sha(Path(args.store) / "config.json")
            or indexconfig["normalization_sha256"] != code_hashes()["text"]
            or retrieval["word"]["source"]["config_sha256"] != indexconfig["store_config_sha256"]
            or retrieval["word"]["source"]["sqlite_sha256"] != sha(Path(args.store) / "records.sqlite3")):
        raise ValueError("Gallery/index/store identity differs")
    if any(r["rows"] != indexconfig["target_records"] for r in retrieval.values()):
        raise ValueError("Sparse and lexical gallery sizes differ")
    modelsha = sha(args.model)
    modelreport = read_json(args.model_report)
    if modelreport["model_sha256"] != modelsha or modelreport["features"] != NAMES:
        raise ValueError("Model report checksum/schema differs")
    model = CatBoostClassifier(thread_count=1)
    model.load_model(args.model)
    if list(model.feature_names_) != NAMES:
        raise ValueError("Expected enhanced76 feature model")
    predict = lambda matrix: model.predict_proba(matrix, thread_count=1)[:, 1]
    neighbors = {c: neighbors_for(retrieval[c]["score_path"], rows) for c in ("word", "char")}
    db = connect(args.store)
    db.execute("PRAGMA cache_size=-8192")
    weights = Weighting(Path(args.index))
    out.mkdir(parents=True)
    matrices, sources = {}, {}
    for cohort in ("tune", "dev"):
        cohortrows = [r for r in rows if r["cohort"] == cohort]
        features, probabilities, manifest = (getattr(args, cohort + "_" + name) for name in ("features", "probabilities", "matrix"))
        baseline, truth, binding = load_baseline(features, probabilities, manifest, cohortrows, args.store, args.index, modelsha)
        if binding["source_config_sha256"] != qm["sources"][cohort]["config_sha256"]:
            raise ValueError("Baseline differs from retrieval comparator")
        targetids = set().union(*(set(baseline[r["entity_id"]]["targets"]) |
            set(neighbors["word"][r["entity_id"]]) | set(neighbors["char"][r["entity_id"]]) for r in cohortrows))
        targets = lookup(db, "records", "entity_id", sorted(targetids))
        if set(targets) != targetids:
            raise ValueError("Missing candidate in provided store")
        master = assemble_master(cohortrows, baseline, neighbors, truth, targets, weights)
        take = master["is_baseline"]
        replay = predict(matrix_for(master, take))
        np.testing.assert_allclose(replay, master["baseline_p"][take], rtol=0, atol=1e-12,
                                   err_msg="Baseline76 replay must reproduce producer probabilities")
        np.savez_compressed(out / f"{cohort}_master.npz", **master)
        matrices[cohort], sources[cohort] = master, {**binding, "baseline_replay_max_abs_error": float(np.max(abs(replay-master["baseline_p"][take]))),
            "master_sha256": sha(out / f"{cohort}_master.npz"), "union_top50_pairs": len(master["y"]),
            "baseline_pairs": int(take.sum())}
        del targets
        print(json.dumps({"prepared": cohort, "pairs": len(master["y"]), "baseline_replay": "passed"}), flush=True)
    db.close()
    policies, grid = tune_policies(matrices["tune"], predict)
    tuning = matrices["tune"]
    baseline_grid = [{"threshold": threshold,
                      **metrics(tuning, tuning["is_baseline"] & (tuning["baseline_p"] >= threshold))}
                     for threshold in UNION_THRESHOLDS]
    baseline_control = max(baseline_grid, key=lambda r: (r["macro_f05"], r["threshold"] == BASE_THRESHOLD, r["threshold"]))
    # Persist selection before any development policy scoring.
    new_json(out / "selected_on_tune.json", {"policies": policies, "grid": grid,
        "baseline_retuned_control": baseline_control, "baseline_threshold_grid": baseline_grid})
    results = {}
    for cohort, a in matrices.items():
        control = a["is_baseline"] & (a["baseline_p"] >= baseline_control["threshold"])
        results[cohort] = {"baseline": metrics(a, base_decisions(a)),
            "baseline_retuned_control": {"threshold": baseline_control["threshold"],
                "metrics": metrics(a, control), "comparison": compare(a, control)}, "policies": {}}
        cache = {}
        selected_arrays = {"baseline": base_decisions(a), "baseline_retuned_control": control}
        for family, policy in policies.items():
            key = (policy["channel"], policy["k"])
            if policy["mode"] == "baseline":
                take = a["is_baseline"]
                chosen = base_decisions(a)
            else:
                if key not in cache:
                    take = recipe_mask(a, *key)
                    cache[key] = take, predict(matrix_for(a, take))
                take, p = cache[key]
                chosen = decisions(a, take, p, policy["mode"], policy["threshold"])
            selected_arrays[family] = chosen
            results[cohort]["policies"][family] = {"policy": policy, "metrics": metrics(a, chosen),
                "candidate_pairs_scored": int(take.sum()), "additional_candidates": int((take & ~a["is_baseline"]).sum()),
                "accepted_new_candidate_tp": int((chosen & ~a["is_baseline"] & (a["y"] == 1)).sum()),
                "accepted_new_candidate_fp": int((chosen & ~a["is_baseline"] & (a["y"] == 0)).sum()),
                "comparison": compare(a, chosen), "comparison_vs_retuned_baseline": compare(a, chosen, control)}
        np.savez_compressed(out / f"{cohort}_decisions.npz", **selected_arrays)
        results[cohort]["decisions_sha256"] = sha(out / f"{cohort}_decisions.npz")
    report = {"complete": True, "diagnostic_only": True, "training": "none", "final_audit_used": False,
        "model_sha256": modelsha, "model_report_sha256": sha(args.model_report),
        "retrieval_evaluation_report_sha256": sha(Path(args.evaluation) / "report.json"),
        "runner_sha256": sha(__file__), "component_sha256": code_hashes(),
        "forward_weighting_index_config_sha256": sha(Path(args.index) / "config.json"),
        "baseline_threshold": BASE_THRESHOLD, "retrieval": retrieval, "sources": sources,
        "selection_sha256": sha(out / "selected_on_tune.json"), "selected_on_tune": policies,
        "baseline_retuned_control_selected_on_tune": baseline_control,
        "results": results, "elapsed_seconds": time.perf_counter()-started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "limitations": ["Zero lexical-hit kNN-only candidates are out-of-training-recipe transfer.",
            "Whole candidate-union context is recomputed; additive policy deliberately preserves original baseline probabilities.",
            "Only14 candidate-retrieval misses on400 tune owners; policy selection is uncertain.",
            "No owner-filter or neural combination, no French quality estimate, not hidden leaderboard evidence."]}
    new_json(out / "report.json", report)
    print(json.dumps({"selected": policies, "results": results, "seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "evaluation", "store", "index", "model", "model-report", "output",
                 "tune-features", "dev-features", "tune-probabilities", "dev-probabilities", "tune-matrix", "dev-matrix"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--galleries", nargs=2, required=True)
    p.add_argument("--scores", nargs=2, required=True)
    run(p.parse_args())
