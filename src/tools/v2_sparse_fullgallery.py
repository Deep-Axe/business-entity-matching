"""Resumable full-gallery, country-blocked hashed TF-IDF retrieval experiment.

Ranking never reads labels. The representation is deliberately NOT the earlier
optimistic 100k-pool vocabulary: nonnegative hashing, corpus-global smoothed IDF,
sublinear TF, and L2 normalization. Top-k is exact in this hashed representation.
Separate `evaluate` joins supplied gold only after retrieval has completed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

import numpy as np
from anyascii import anyascii
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from ber.data import read_tsv, write_json, write_tsv
from ber.store import connect
from ber.text import normalize
import ber.text


CHANNELS = {"word": ("word", (1, 2)), "char": ("char_wb", (3, 4))}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def new_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temporary = Path(f.name)
    os.link(temporary, path)
    temporary.unlink()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_array_once(path, array):
    """Atomic, no-overwrite global output; identical interrupted finalization can resume."""
    path = Path(path)
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), array)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
        temporary = Path(f.name)
    # Link is atomic and rejects collisions; only our newly created scratch is unlinked.
    os.link(temporary, path)
    temporary.unlink()


def representation(channel, bins):
    if channel not in CHANNELS or bins < 16 or bins & (bins - 1):
        raise ValueError("channel must be word/char and bins a power of two >=16")
    analyzer, grams = CHANNELS[channel]
    return {"channel": channel, "analyzer": analyzer, "ngram_range": list(grams),
            "bins": bins, "alternate_sign": False, "sublinear_tf": True,
            "idf": "log((1+N)/(1+df))+1", "norm": "l2", "lowercase": False,
            "romanization": "anyascii then ber.text.normalize, name plus address",
            "normalizer_sha256": sha(ber.text.__file__),
            "versions": {p: importlib.metadata.version(p) for p in
                         ("numpy", "scipy", "scikit-learn", "anyascii")}}


def vectorizer(rep):
    return HashingVectorizer(n_features=rep["bins"], analyzer=rep["analyzer"],
        ngram_range=tuple(rep["ngram_range"]), alternate_sign=False, norm=None,
        lowercase=False, dtype=np.float32)


def texts(rows):
    return [normalize(anyascii(r["business_name"])) + " " +
            normalize(anyascii(r["business_address"]), True) for r in rows]


def counts_for(rows, rep):
    x = vectorizer(rep).transform(texts(rows)).tocsr()
    x.sum_duplicates()
    x.sort_indices()
    if x.nnz and np.min(x.data) <= 0:
        raise ValueError("Nonnegative count hashing invariant violated")
    return x


def df_for(x):
    # CSR has one entry per document/hash bucket, not one per token occurrence.
    return np.unique(x.indices, return_counts=True)


def tfidf(x, idf):
    x = x.copy().astype(np.float32)
    np.log(x.data, out=x.data)
    x.data += 1
    x.data *= idf[x.indices]
    return l2_normalize(x, norm="l2", copy=False)


def topk(ids, scores, k):
    """Positive top-k, deterministic lexical entity-ID tie break including kth ties."""
    ids, scores = np.asarray(ids), np.asarray(scores, dtype=np.float32)
    mask = np.isfinite(scores) & (scores > 0) & (ids != "")
    ids, scores = ids[mask], scores[mask]
    if len(scores) > k:
        cutoff = np.partition(scores, -k)[-k]
        keep = scores >= cutoff
        ids, scores = ids[keep], scores[keep]
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def block_topk(qmatrix, qcountries, matrix, ids, countries, k, query_batch=16):
    """Never constructs query x gallery; at most query_batch x block(country)."""
    result_ids = np.full((len(qcountries), k), "", dtype=ids.dtype)
    result_scores = np.zeros((len(qcountries), k), dtype=np.float32)
    for country in sorted(set(qcountries)):
        qi = np.flatnonzero(qcountries == country)
        ti = np.flatnonzero(countries == country)
        if not len(ti):
            continue
        # CSR @ CSR avoids SciPy repeatedly converting a 2^22-row transpose
        # and allocating its indptr for every small query batch.
        transposed = matrix[ti].T.tocsr()
        for start in range(0, len(qi), query_batch):
            batch = qi[start:start + query_batch]
            scores = (qmatrix[batch] @ transposed).tocsr()
            for j, query in enumerate(batch):
                a, b = scores.indptr[j:j+2]
                chosen, values = topk(ids[ti[scores.indices[a:b]]], scores.data[a:b], k)
                result_ids[query, :len(chosen)] = chosen
                result_scores[query, :len(values)] = values
    return result_ids, result_scores


def check_commit(folder, expected=None):
    done = read_json(folder / "done.json")
    if expected is not None:
        for key, value in expected.items():
            if done.get(key) != value:
                raise ValueError(f"Committed metadata mismatch {folder}: {key}")
    for name, digest in done["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Committed checksum mismatch: {folder / name}")
    return done


def commit_stage(stage, destination, meta, names):
    meta = {**meta, "files": {name: sha(stage / name) for name in names}}
    new_json(stage / "done.json", meta)
    if destination.exists():
        raise FileExistsError(destination)
    stage.rename(destination)
    return meta


def prepare(args):
    output = Path(args.output)
    if args.queries_per_split < 400 and not args.allow_small_test:
        raise ValueError("At least 400 tune and 400 dev queries are required")
    rows, sources = [], {}
    seen = set()
    for cohort, path in (("tune", args.tune_features), ("dev", args.dev_features)):
        root = Path(path)
        config = read_json(root / "config.json")
        if config.get("gallery_fold") is not None or config.get("gate_terms") != 0:
            raise ValueError("Comparator must be gate0 full gallery")
        if not read_json(root / "manifest.json").get("complete"):
            raise ValueError("Comparator cache incomplete")
        if "audit" in str(root).lower():
            raise ValueError("Final audit is reserved")
        selected, commits = [], []
        for folder in sorted((root / "shards").iterdir()):
            if not folder.name.isdigit() or len(selected) >= args.queries_per_split:
                continue
            done = read_json(folder / "done.json")
            if sha(folder / "references.tsv") != done["output_sha256"]["references.tsv"]:
                raise ValueError("Reference checksum mismatch")
            commits.append({"shard": folder.name, "done_sha256": sha(folder / "done.json")})
            for row in read_tsv(folder / "references.tsv"):
                if len(selected) >= args.queries_per_split:
                    break
                if row["entity_id"] in seen:
                    raise ValueError("Duplicate query / overlapping cohorts")
                seen.add(row["entity_id"])
                selected.append({**row, "cohort": cohort})
        if len(selected) != args.queries_per_split:
            raise ValueError("Not enough fixed cohort queries")
        rows.extend(selected)
        sources[cohort] = {"path": str(root.resolve()), "config_sha256": sha(root / "config.json"),
                           "shards": commits}
    output.mkdir(parents=True, exist_ok=False)
    write_tsv(output / "queries.tsv", rows,
              ["entity_id", "business_name", "business_address", "country", "cohort"])
    new_json(output / "manifest.json", {"complete": True, "queries": len(rows),
        "queries_sha256": sha(output / "queries.tsv"), "sources": sources,
        "ranking_labels_read": False, "selection": "first fixed N per existing tune/dev cohort"})
    print(json.dumps({"prepared": str(output), "queries": len(rows)}), flush=True)


def build_worker(task):
    output, block_id, rows, rep, config_hash, minimum_free_gib = task
    started = time.perf_counter()
    root = Path(output) / "blocks"
    if shutil.disk_usage(root).free < minimum_free_gib * 1024**3:
        raise RuntimeError("Disk guard: insufficient free space to commit another count block")
    stage = Path(tempfile.mkdtemp(prefix=f".{block_id:06d}.", dir=root))
    matrix = counts_for(rows, rep)
    sparse.save_npz(stage / "counts.npz", matrix, compressed=True)
    indices, counts = df_for(matrix)
    np.savez_compressed(stage / "df.npz", indices=indices.astype(np.int32), counts=counts.astype(np.uint32))
    np.savez_compressed(stage / "targets.npz", ids=np.asarray([r["entity_id"] for r in rows]),
                        countries=np.asarray([r["country"] for r in rows]))
    meta = {"block": block_id, "rows": len(rows), "first_row_id": rows[0]["row_id"],
            "last_row_id": rows[-1]["row_id"], "input_rows_sha256": object_sha(rows),
            "config_sha256": config_hash, "nnz": matrix.nnz,
            "seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    return commit_stage(stage, root / f"{block_id:06d}", meta,
                        ["counts.npz", "df.npz", "targets.npz"])


def ordered_parallel(fn, tasks, workers):
    if workers == 1:
        yield from map(fn, tasks)
        return
    # Bounded submission: executor.map eagerly queues the whole gallery iterator.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        for task in tasks:
            pending.append(pool.submit(fn, task))
            if len(pending) >= workers:
                yield pending.pop(0).result()
        for future in pending:
            yield future.result()


def source_fingerprint(store):
    store = Path(store)
    wal = store / "records.sqlite3-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source has pending WAL writes; require immutable checkpointed store")
    if not read_json(store / "manifest.json").get("complete"):
        raise ValueError("Training store incomplete")
    if read_json(store / "config.json").get("split") != "train":
        raise ValueError("This experiment requires the full provided TRAIN target gallery")
    return {"config_sha256": sha(store / "config.json"),
            "sqlite_sha256": sha(store / "records.sqlite3")}


def build(args):
    started = time.perf_counter()
    root, store = Path(args.output), Path(args.store)
    source = source_fingerprint(store)
    db = connect(store)
    db.execute("PRAGMA cache_size=-8192")
    n, first, last = db.execute("SELECT COUNT(*),MIN(row_id),MAX(row_id) FROM records").fetchone()
    if first != 1 or last != n:
        raise ValueError("Gallery row IDs must cover contiguous 1..N exactly")
    requested = min(args.limit_targets, n) if args.limit_targets else n
    config = {"version": 1, "representation": representation(args.channel, args.bins),
              "source": source, "total_gallery_rows": n, "requested_rows": requested,
              "full_gallery": requested == n, "block_size": args.block_size,
              "runner_sha256": sha(__file__), "all_target_folds": True,
              "labels_read": False, "country_blocked": True}
    if root.exists():
        if not args.resume:
            raise FileExistsError(root)
        if read_json(root / "config.json") != config:
            raise ValueError("Resume configuration/source/code checksum mismatch")
    else:
        root.mkdir(parents=True)
        (root / "blocks").mkdir()
        new_json(root / "config.json", config)
    config_hash = sha(root / "config.json")
    block_count = math.ceil(requested / args.block_size)
    if any(p.name.isdigit() and int(p.name) >= block_count for p in (root / "blocks").iterdir()):
        raise ValueError("Unexpected committed block outside requested gallery")
    progress = {"stage": "build", "total_blocks": block_count, "committed_blocks": 0}

    def tasks():
        scheduled = 0
        for block_id in range(block_count):
            begin = block_id * args.block_size + 1
            end = min((block_id + 1) * args.block_size, requested)
            folder = root / "blocks" / f"{block_id:06d}"
            rows = [{"row_id": r["row_id"], "entity_id": r["entity_id"],
                     "business_name": r["name"], "business_address": r["address"], "country": r["country"]}
                    for r in db.execute("SELECT row_id,entity_id,name,address,country FROM records "
                                        "WHERE row_id BETWEEN ? AND ? ORDER BY row_id", (begin, end))]
            if len(rows) != end - begin + 1:
                raise ValueError("Missing gallery rows")
            if folder.exists():
                check_commit(folder, {"block": block_id, "first_row_id": begin, "last_row_id": end,
                    "rows": len(rows), "input_rows_sha256": object_sha(rows), "config_sha256": config_hash})
                progress["committed_blocks"] += 1
            else:
                if args.stop_after_blocks and scheduled >= args.stop_after_blocks:
                    break
                scheduled += 1
                yield (str(root), block_id, rows, config["representation"], config_hash,
                       getattr(args, "minimum_free_gib", 6))

    for done in ordered_parallel(build_worker, tasks(), args.workers):
        progress["committed_blocks"] += 1
        progress.update(last_block=done["block"], elapsed_seconds=time.perf_counter() - started,
                        last_block_seconds=done["seconds"], worker_peak_rss_mib=done["peak_rss_mib"])
        write_json(root / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    db.close()
    if progress["committed_blocks"] != block_count:
        write_json(root / "progress.json", {**progress, "stage": "build_paused", "complete": False})
        print(json.dumps({"paused": True, "committed_blocks": progress["committed_blocks"],
                          "total_blocks": block_count, "elapsed_seconds": time.perf_counter() - started}), flush=True)
        return
    dfs = np.zeros(args.bins, dtype=np.uint64)
    commits, count, nnz = [], 0, 0
    for block_id in range(block_count):
        folder = root / "blocks" / f"{block_id:06d}"
        done = check_commit(folder, {"block": block_id, "config_sha256": config_hash})
        count += done["rows"]
        nnz += done["nnz"]
        with np.load(folder / "df.npz", allow_pickle=False) as z:
            dfs[z["indices"]] += z["counts"]
        commits.append({"block": block_id, "done_sha256": sha(folder / "done.json")})
    if count != requested or dfs.max() > requested:
        raise ValueError("Incomplete gallery or invalid document frequencies")
    idf = (np.log((1 + requested) / (1 + dfs.astype(np.float64))) + 1).astype(np.float32)
    # A complete resume verifies existing global outputs rather than overwriting.
    if (root / "manifest.json").exists():
        old = read_json(root / "manifest.json")
        if (old["blocks"] != commits or sha(root / "idf.npy") != old["idf_sha256"]
                or sha(root / "df.npy") != old["df_sha256"]):
            raise ValueError("Completed gallery manifest/IDF changed")
        np.testing.assert_array_equal(np.load(root / "idf.npy"), idf)
        return
    save_array_once(root / "idf.npy", idf)
    save_array_once(root / "df.npy", dfs)
    new_json(root / "manifest.json", {"complete": True, "full_gallery": requested == n,
        "rows": requested, "nnz": nnz, "config_sha256": config_hash,
        "idf_sha256": sha(root / "idf.npy"), "df_sha256": sha(root / "df.npy"),
        "blocks": commits, "elapsed_seconds": time.perf_counter() - started})
    write_json(root / "progress.json", {**progress, "stage": "built", "complete": True})


_SCORE_STATE = None


def score_init(gallery, queries, k, query_batch):
    global _SCORE_STATE
    gallery = Path(gallery)
    config = read_json(gallery / "config.json")
    rows = list(read_tsv(Path(queries) / "queries.tsv"))
    idf = np.load(gallery / "idf.npy", mmap_mode="r")
    qmatrix = tfidf(counts_for(rows, config["representation"]), idf)
    _SCORE_STATE = (gallery, qmatrix, np.asarray([r["country"] for r in rows]), idf, k, query_batch)


def score_worker(task):
    output, block_id, source_done_hash, config_hash = task
    gallery, queries, countries, idf, k, query_batch = _SCORE_STATE
    started = time.perf_counter()
    source = gallery / "blocks" / f"{block_id:06d}"
    if sha(source / "done.json") != source_done_hash:
        raise ValueError("Gallery commit changed")
    check_commit(source)
    matrix = tfidf(sparse.load_npz(source / "counts.npz"), idf)
    with np.load(source / "targets.npz", allow_pickle=False) as z:
        ids, scores = block_topk(queries, countries, matrix, z["ids"], z["countries"], k, query_batch)
    root = Path(output) / "blocks"
    stage = Path(tempfile.mkdtemp(prefix=f".{block_id:06d}.", dir=root))
    np.savez_compressed(stage / "topk.npz", ids=ids, scores=scores)
    return commit_stage(stage, root / f"{block_id:06d}",
        {"block": block_id, "source_done_sha256": source_done_hash, "config_sha256": config_hash,
         "seconds": time.perf_counter() - started,
         "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}, ["topk.npz"])


def score(args):
    started = time.perf_counter()
    gallery, querydir, root = Path(args.gallery), Path(args.queries), Path(args.output)
    gc, gm = read_json(gallery / "config.json"), read_json(gallery / "manifest.json")
    qm = read_json(querydir / "manifest.json")
    if not gm["complete"] or not qm["complete"]:
        raise ValueError("Incomplete gallery/queries")
    if not gm["full_gallery"] and not args.allow_partial_gallery:
        raise ValueError("Partial gallery is a timing diagnostic only; explicit opt-in required")
    if sha(gallery / "config.json") != gm["config_sha256"] or sha(gallery / "idf.npy") != gm["idf_sha256"]:
        raise ValueError("Gallery config/IDF checksum changed")
    if sha(querydir / "queries.tsv") != qm["queries_sha256"]:
        raise ValueError("Query checksum changed")
    if gc["representation"] != representation(gc["representation"]["channel"], gc["representation"]["bins"]):
        raise ValueError("Representation dependencies changed")
    config = {"gallery_manifest_sha256": sha(gallery / "manifest.json"),
              "queries_manifest_sha256": sha(querydir / "manifest.json"),
              "queries_sha256": qm["queries_sha256"], "channel": gc["representation"]["channel"],
              "k": args.k, "query_batch": args.query_batch, "runner_sha256": sha(__file__),
              "full_gallery": gm["full_gallery"], "labels_read": False}
    if root.exists():
        if not args.resume:
            raise FileExistsError(root)
        if read_json(root / "config.json") != config:
            raise ValueError("Score resume configuration mismatch")
    else:
        root.mkdir(parents=True)
        (root / "blocks").mkdir()
        new_json(root / "config.json", config)
    config_hash = sha(root / "config.json")
    valid_ids = {b["block"] for b in gm["blocks"]}
    if valid_ids != set(range(math.ceil(gm["rows"] / gc["block_size"]))):
        raise ValueError("Gallery manifest lacks exact block coverage")
    if any(p.name.isdigit() and int(p.name) not in valid_ids for p in (root / "blocks").iterdir()):
        raise ValueError("Unexpected scored block")
    tasks, completed = [], 0
    for block in gm["blocks"]:
        i = block["block"]
        folder = root / "blocks" / f"{i:06d}"
        if folder.exists():
            check_commit(folder, {"block": i, "source_done_sha256": block["done_sha256"],
                                  "config_sha256": config_hash})
            completed += 1
        else:
            tasks.append((str(root), i, block["done_sha256"], config_hash))
    # Initializer shares only readonly query vectors/IDF within each bounded worker.
    if args.workers == 1:
        score_init(str(gallery), str(querydir), args.k, args.query_batch)
        results = map(score_worker, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=score_init,
                  initargs=(str(gallery), str(querydir), args.k, args.query_batch))
        results = pool.map(score_worker, tasks, chunksize=1)  # tasks are small metadata only
    try:
        for done in results:
            completed += 1
            progress = {"stage": "score", "committed_blocks": completed, "total_blocks": len(gm["blocks"]),
                        "elapsed_seconds": time.perf_counter() - started, "last_block_seconds": done["seconds"],
                        "worker_peak_rss_mib": done["peak_rss_mib"]}
            write_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    finally:
        if pool:
            pool.shutdown()
    rows = list(read_tsv(querydir / "queries.tsv"))
    best_ids = [np.array([], dtype="U1") for _ in rows]
    best_scores = [np.array([], dtype=np.float32) for _ in rows]
    commits = []
    for block in gm["blocks"]:
        folder = root / "blocks" / f"{block['block']:06d}"
        check_commit(folder, {"config_sha256": config_hash, "source_done_sha256": block["done_sha256"]})
        with np.load(folder / "topk.npz", allow_pickle=False) as z:
            block_ids, block_scores = z["ids"], z["scores"]
            for q in range(len(rows)):
                best_ids[q], best_scores[q] = topk(np.concatenate((best_ids[q], block_ids[q])),
                    np.concatenate((best_scores[q], block_scores[q])), args.k)
        commits.append({"block": block["block"], "done_sha256": sha(folder / "done.json")})
    if (root / "manifest.json").exists():
        old = read_json(root / "manifest.json")
        if old["blocks"] != commits or sha(root / "neighbors.tsv") != old["neighbors_sha256"]:
            raise ValueError("Completed score output changed")
        return
    fd, name = tempfile.mkstemp(prefix=".neighbors.tsv.", dir=root)
    os.close(fd)
    temporary = Path(name)
    write_tsv(temporary, ({"source1_entity_id": row["entity_id"], "cohort": row["cohort"],
        "country": row["country"], "target_id": target, "rank": j + 1,
        "score": format(float(best_scores[q][j]), ".9g"), "channel": config["channel"]}
        for q, row in enumerate(rows) for j, target in enumerate(best_ids[q])),
        ["source1_entity_id", "cohort", "country", "target_id", "rank", "score", "channel"])
    if (root / "neighbors.tsv").exists():
        if sha(root / "neighbors.tsv") != sha(temporary):
            raise ValueError("Unfinalized neighbor output differs; refusing overwrite")
    else:
        os.link(temporary, root / "neighbors.tsv")
    temporary.unlink()
    new_json(root / "manifest.json", {"complete": True, "full_gallery": gm["full_gallery"],
        "config_sha256": config_hash, "neighbors_sha256": sha(root / "neighbors.tsv"),
        "blocks": commits, "queries": len(rows), "gallery_rows": gm["rows"],
        "elapsed_seconds": time.perf_counter() - started, "labels_read": False})
    write_json(root / "progress.json", {"stage": "scored", "complete": True, "committed_blocks": completed})


def evaluate(args):
    querydir, out = Path(args.queries), Path(args.output)
    qm = read_json(querydir / "manifest.json")
    if sha(querydir / "queries.tsv") != qm["queries_sha256"]:
        raise ValueError("Queries changed")
    rows = list(read_tsv(querydir / "queries.tsv"))
    wanted = {r["entity_id"] for r in rows}
    channels, provenance = {}, {}
    for path in args.scores:
        path = Path(path)
        config, manifest = read_json(path / "config.json"), read_json(path / "manifest.json")
        if not manifest["complete"] or not manifest["full_gallery"]:
            raise ValueError("Evaluation requires complete genuinely full galleries")
        if config["queries_sha256"] != qm["queries_sha256"] or sha(path / "neighbors.tsv") != manifest["neighbors_sha256"]:
            raise ValueError("Score/query alignment checksum mismatch")
        channel = config["channel"]
        if channel in channels:
            raise ValueError("Duplicate retrieval channel")
        neighbors = {r: [] for r in wanted}
        for row in read_tsv(path / "neighbors.tsv"):
            r = row["source1_entity_id"]
            if r not in wanted or int(row["rank"]) != len(neighbors[r]) + 1:
                raise ValueError("Unexpected query or noncontiguous ranking")
            neighbors[r].append(row["target_id"])
        channels[channel] = neighbors
        provenance[channel] = {"manifest_sha256": sha(path / "manifest.json"), "gallery_rows": manifest["gallery_rows"]}
    gold, base = {}, {}
    for cohort, override in (("tune", args.tune_features), ("dev", args.dev_features)):
        source = qm["sources"][cohort]
        path = Path(override or source["path"])
        if sha(path / "config.json") != source["config_sha256"]:
            raise ValueError("Comparator changed")
        for shard in source["shards"]:
            folder = path / "shards" / shard["shard"]
            if sha(folder / "done.json") != shard["done_sha256"]:
                raise ValueError("Comparator commit changed")
            done = read_json(folder / "done.json")
            for name in ("features.npz", "ground_truth.tsv"):
                if sha(folder / name) != done["output_sha256"][name]:
                    raise ValueError("Comparator data changed")
            with np.load(folder / "features.npz", allow_pickle=False) as z:
                for q, r in enumerate(z["reference_ids"]):
                    if str(r) in wanted:
                        base[str(r)] = set(map(str, z["target_ids"][z["query_ids"] == q]))
            for row in read_tsv(folder / "ground_truth.tsv"):
                if row["source1_entity_id"] in wanted:
                    gold[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    if set(gold) != wanted or set(base) != wanted:
        raise ValueError("Comparator/gold query coverage mismatch")
    report = {"purpose": "retrieval recall only, not final classifier or submission score",
              "provenance": provenance, "queries_sha256": qm["queries_sha256"],
              "representation": "hashed TF-IDF; collisions possible; not prior vocabulary pilot",
              "selection": "No dev policy selection; report prespecified top10/20/50 per channel and union",
              "splits": {}}
    for cohort in ("tune", "dev"):
        for country in ("all", "India", "US"):
            ids = [r["entity_id"] for r in rows if r["cohort"] == cohort and (country == "all" or r["country"] == country)]
            stats = {"queries": len(ids), "gold_pairs": sum(len(gold[r]) for r in ids),
                     "gate0_pairs": sum(len(base[r]) for r in ids),
                     "gate0_gold": sum(len(base[r] & gold[r]) for r in ids), "retrieval": {}}
            for channel in [*channels, "union"]:
                for k in (10, 20, 50):
                    chosen = {r: set().union(*(set(v[r][:k]) for v in channels.values())) if channel == "union"
                              else set(channels[channel][r][:k]) for r in ids}
                    stats["retrieval"][f"{channel}_top{k}"] = {
                        "new_candidates": sum(len(chosen[r] - base[r]) for r in ids),
                        "additional_gold": sum(len((chosen[r] - base[r]) & gold[r]) for r in ids),
                        "gold_after_union": sum(len((chosen[r] | base[r]) & gold[r]) for r in ids)}
            report["splits"][f"{cohort}_{country}"] = stats
    out.mkdir(parents=True, exist_ok=False)
    for k in (10, 20, 50):
        write_tsv(out / f"candidate_union_top{k}.tsv", ({"source1_entity_id": r["entity_id"],
            "cohort": r["cohort"], "candidate_entity_ids": ",".join(sorted(base[r["entity_id"]] |
                set().union(*(set(v[r["entity_id"]][:k]) for v in channels.values()))))} for r in rows),
            ["source1_entity_id", "cohort", "candidate_entity_ids"])
        write_tsv(out / f"candidate_additions_top{k}.tsv", ({"source1_entity_id": r["entity_id"],
            "cohort": r["cohort"], "candidate_entity_ids": ",".join(sorted(
                set().union(*(set(v[r["entity_id"]][:k]) for v in channels.values())) - base[r["entity_id"]]))}
            for r in rows), ["source1_entity_id", "cohort", "candidate_entity_ids"])
    new_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("prepare")
    q.add_argument("--tune-features", required=True)
    q.add_argument("--dev-features", required=True)
    q.add_argument("--queries-per-split", type=int, default=400)
    q.add_argument("--allow-small-test", action="store_true")
    q.add_argument("--output", required=True)
    q.set_defaults(fn=prepare)
    b = commands.add_parser("build")
    b.add_argument("--store", required=True)
    b.add_argument("--channel", choices=CHANNELS, required=True)
    b.add_argument("--bins", type=int, default=2**22)
    b.add_argument("--block-size", type=int, default=10000)
    b.add_argument("--limit-targets", type=int, help="Timing diagnostic only, never full evaluation")
    b.add_argument("--stop-after-blocks", type=int, help="Commit at most N new blocks, then pause safely")
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--minimum-free-gib", type=float, default=6)
    b.add_argument("--resume", action="store_true")
    b.add_argument("--output", required=True)
    b.set_defaults(fn=build)
    s = commands.add_parser("score")
    s.add_argument("--gallery", required=True)
    s.add_argument("--queries", required=True)
    s.add_argument("--k", type=int, default=50)
    s.add_argument("--query-batch", type=int, default=16)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--allow-partial-gallery", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--output", required=True)
    s.set_defaults(fn=score)
    e = commands.add_parser("evaluate")
    e.add_argument("--queries", required=True)
    e.add_argument("--scores", nargs="+", required=True)
    e.add_argument("--tune-features")
    e.add_argument("--dev-features")
    e.add_argument("--output", required=True)
    e.set_defaults(fn=evaluate)
    args = p.parse_args()
    for name in ("workers", "k", "query_batch", "block_size", "limit_targets", "queries_per_split", "stop_after_blocks"):
        if getattr(args, name, None) is not None and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if getattr(args, "minimum_free_gib", 6) < 0:
        p.error("minimum_free_gib must be nonnegative")
    args.fn(args)


if __name__ == "__main__":
    cli()
