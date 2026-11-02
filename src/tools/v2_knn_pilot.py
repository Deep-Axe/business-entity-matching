"""Bounded, optimistic sparse-neighbor feasibility screen; never a submission.

Only provided training/tuning data are read. Labels construct the target pool and
measure results, never ranking. Full-gallery validation remains mandatory.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from anyascii import anyascii
from catboost import CatBoostClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.context_features import model_matrix
from ber.data import read_tsv, write_json, write_tsv
from ber.store import connect, lookup, record_row
from ber.text import Record, normalize


def sparse_topk(scores, countries, country, k):
    """Exact positive-score top-k with deterministic index tie break, no dense row."""
    row = scores.tocsr()
    mask = (countries[row.indices] == country) & (row.data > 0)
    ids, values = row.indices[mask], row.data[mask]
    order = np.lexsort((ids, -values))[:k]
    return ids[order].tolist()


def flags(left, right):
    """Overlapping descriptive flags, not adjudicated root causes."""
    result = []
    if not left.address or not right.address:
        result.append("missing_address")
    if left.numbers and right.numbers and not left.numbers & right.numbers:
        result.append("disjoint_address_numbers")
    if any(ord(c) > 127 for c in left.original_name + right.original_name):
        result.append("nonascii_name")
    if not left.name_tokens & right.name_tokens:
        result.append("no_shared_unicode_name_token")
    if not set(left.ascii_name.split()) & set(right.ascii_name.split()):
        result.append("no_shared_romanized_name_token")
    if left.name == right.name:
        result.append("exact_name")
    if left.address and left.address == right.address:
        result.append("exact_address")
    if set(left.ascii_name.split()) == set(right.ascii_name.split()):
        result.append("same_romanized_name_tokens")
    return result


def main(args):
    started = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(args.features)
    config = json.loads((cache / "config.json").read_text())
    if config["gallery_fold"] is not None:
        raise ValueError("Lexical comparator must use full gallery")
    if "audit" in str(cache).lower():
        raise ValueError("Reserved audit is not a tuning source")
    model = CatBoostClassifier(thread_count=1)
    model.load_model(str(Path(args.model) / "model.cbm"))
    ref_ids, lexical, predicted, gold = [], {}, {}, {}
    for folder in sorted((cache / "shards").iterdir()):
        if len(ref_ids) >= args.queries:
            break
        with np.load(folder / "features.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        probabilities = model.predict_proba(
            model_matrix(arrays, model.feature_names_), thread_count=1)[:, 1]
        truth = {r["source1_entity_id"]: set(filter(None, r["matched_entity_ids"].split(",")))
                 for r in read_tsv(folder / "ground_truth.tsv")}
        for q, rid in enumerate(arrays["reference_ids"]):
            if len(ref_ids) >= args.queries:
                break
            rid = str(rid)
            ref_ids.append(rid)
            selection = arrays["query_ids"] == q
            lexical[rid] = set(map(str, arrays["target_ids"][selection]))
            predicted[rid] = set(map(str, arrays["target_ids"][selection & (probabilities >= .74)]))
            gold[rid] = truth[rid]
    del model, arrays, probabilities
    gc.collect()
    db = connect(args.store)
    db.execute("PRAGMA cache_size=-8192")
    anchors = lookup(db, "anchors", "entity_id", ref_ids)
    if any(r["fold"] != "tune" for r in anchors.values()):
        raise ValueError("This feasibility screen is tuning-only")
    mandatory = set().union(*lexical.values(), *gold.values())
    target_rows = lookup(db, "records", "entity_id", sorted(mandatory))
    if len(target_rows) != len(mandatory) or len(mandatory) > args.targets:
        raise ValueError("Missing targets or pool limit smaller than mandatory pool")
    max_id = db.execute("SELECT MAX(row_id) FROM records").fetchone()[0]
    rng = np.random.default_rng(2926)
    while len(target_rows) < args.targets:
        need = args.targets - len(target_rows)
        ids = list(map(int, rng.integers(1, max_id + 1, size=min(need + 100, 5000))))
        for r in lookup(db, "records", "row_id", ids).values():
            if len(target_rows) < args.targets:
                target_rows[r["entity_id"]] = r
    db.close()
    ids = sorted(target_rows)
    countries = np.array([target_rows[e]["country"] for e in ids])
    query_records = {e: Record.from_row(record_row(anchors[e])) for e in ref_ids}
    # Pool membership uses labels; text representations and ranking never do.
    target_names = [normalize(anyascii(target_rows[e]["name"])) for e in ids]
    target_text = [n + " " + normalize(anyascii(target_rows[e]["address"]), True)
                   for e, n in zip(ids, target_names)]
    query_names = [query_records[e].ascii_name for e in ref_ids]
    query_text = [query_records[e].ascii_name + " " + query_records[e].ascii_address for e in ref_ids]
    report = {
        "purpose": "optimistic bounded-pool feasibility, not full-gallery/model accuracy",
        "queries": len(ref_ids), "pool_targets": len(ids), "mandatory_targets": len(mandatory),
        "query_fold": "tune", "query_ids_sha256": hashlib.sha256("\n".join(ref_ids).encode()).hexdigest(),
        "pool_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "countries": dict(Counter(anchors[e]["country"] for e in ref_ids)),
        "pool_countries": dict(Counter(countries)), "seed": 2926,
        "baseline_threshold": .74, "baseline_pairs": sum(map(len, lexical.values())),
        "gold_pairs": sum(map(len, gold.values())),
        "baseline_gold_retrieved": sum(len(gold[e] & lexical[e]) for e in ref_ids),
        "caveats": ["All positives and baseline hard candidates forced into 100k pool; recall optimistic",
                    "TF-IDF fitted on bounded pool, not full target corpus",
                    "Unknown/unlabeled matches treated as negatives under supplied gold",
                    "Tuning-only choices; full-gallery and separate development evaluation required",
                    "No French labels; no French quality estimate",
                    "Sparse retrieval only; no new matcher trained or submission changed"],
        "channels": {},
    }
    all_neighbors = {e: set() for e in ref_ids}
    for channel, analyzer, grams, texts, queries in [
        ("word_combined", "word", (1, 2), target_text, query_text),
        ("char_name", "char_wb", (3, 4), target_names, query_names),
        ("char_combined", "char_wb", (3, 4), target_text, query_text),
    ]:
        then = time.perf_counter()
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=grams,
            min_df=1, max_features=100000, sublinear_tf=True, dtype=np.float32,
            lowercase=False, norm="l2")
        matrix = vectorizer.fit_transform(texts).tocsr()
        query_matrix = vectorizer.transform(queries).tocsr()
        transpose = matrix.T.tocsc()
        built = time.perf_counter()
        stats = {k: Counter() for k in (10, 20, 50)}
        hits = []
        for i, e in enumerate(ref_ids):
            scores = query_matrix[i] @ transpose
            ranked = [ids[j] for j in sparse_topk(scores, countries, anchors[e]["country"], 50)]
            all_neighbors[e].update(ranked[:20])
            for k in stats:
                chosen = set(ranked[:k])
                new = chosen - lexical[e]
                stats[k].update(pairs=len(chosen), recovered_gold=len(chosen & gold[e]),
                    new_pairs=len(new), new_gold=len(new & gold[e]), new_negative=len(new - gold[e]))
            hits.extend({"source1_entity_id": e, "target_id": t, "rank": j + 1,
                         "is_gold": int(t in gold[e]), "new_vs_lexical": int(t not in lexical[e])}
                        for j, t in enumerate(ranked))
        write_tsv(out / f"{channel}_neighbors.tsv", hits, list(hits[0]))
        report["channels"][channel] = {"k": stats, "vocabulary": len(vectorizer.vocabulary_),
            "matrix_nnz": int(matrix.nnz), "fit_seconds": built - then,
            "query_seconds": time.perf_counter() - built}
        print(json.dumps({"channel": channel, **report["channels"][channel]}), flush=True)
        del vectorizer, matrix, transpose, query_matrix, hits, scores
        gc.collect()
    report["union_top20"] = {
        "new_pairs": sum(len(all_neighbors[e] - lexical[e]) for e in ref_ids),
        "new_gold": sum(len((all_neighbors[e] - lexical[e]) & gold[e]) for e in ref_ids),
        "gold_after_union": sum(len((all_neighbors[e] | lexical[e]) & gold[e]) for e in ref_ids),
    }
    taxonomy, examples = {}, []
    for e in ref_ids:
        groups = {"retrieval_miss": gold[e] - lexical[e],
                  "classifier_rejection": (gold[e] & lexical[e]) - predicted[e],
                  "false_acceptance": predicted[e] - gold[e]}
        for kind, targets in groups.items():
            stats = taxonomy.setdefault(kind, Counter())
            for t in sorted(targets):
                target = Record.from_row(record_row(target_rows[t]))
                pair_flags = flags(query_records[e], target)
                stats.update(["pairs", "country:" + anchors[e]["country"], "source:" + t[:2], *pair_flags])
                examples.append({"source1_entity_id": e, "target_id": t, "error": kind,
                    "country": anchors[e]["country"], "flags": ",".join(pair_flags),
                    "knn_recovered_top20": int(t in all_neighbors[e]),
                    "reference_name": anchors[e]["name"], "reference_address": anchors[e]["address"],
                    "target_name": target_rows[t]["name"], "target_address": target_rows[t]["address"]})
    report["overlapping_error_flags"] = taxonomy
    if examples:
        write_tsv(out / "error_examples.tsv", examples, list(examples[0]))
    write_tsv(out / "references.tsv", ({"entity_id": e} for e in ref_ids), ["entity_id"])
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="artifacts/deadline_features_full_tune_v1")
    p.add_argument("--store", default="artifacts/train_store_v2")
    p.add_argument("--model", default="artifacts/deadline_model_50k_v1")
    p.add_argument("--queries", type=int, default=400)
    p.add_argument("--targets", type=int, default=100000)
    p.add_argument("--output", required=True)
    main(p.parse_args())
