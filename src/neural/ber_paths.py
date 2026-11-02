"""Resolvable locations. No machine-absolute paths.

Set these before running any script (defaults assume the repo root is cwd):

  BER_ROOT     repository root (default: current working directory)
  BER_DATA     dataset folder with train/ and test/ TSVs
  CE_DATA      working directory for pair pickles and scores (default: $BER_ROOT/work/ce)
  CE_OUT       local retrain checkpoint parent (default: same as CE_DATA; published weights are BER_WEIGHTS)
  CE_PREP      prepared_gate0_full_{dev,tune}_v1 parent (default: model/prep when present)
  BER_OWNER    folder with policy.json and text_scorer.json
  BER_WEIGHTS  folder with encoder model.pt files (e5_cont1300k/, kA/ … kG/)
  BER_HF_REPO  Hugging Face repo with those files (default: DeeAxe/business-entity-matching-encoders)
  BER_E5       HuggingFace id or local dir for multilingual-e5-small
  BER_E5_BASE  HuggingFace id or local dir for multilingual-e5-base
  BER_E5_REVISION / BER_E5_BASE_REVISION  immutable Hub commits for those ids
  BER_STORE    parent of {train,test}_store_v2 (default: $BER_ROOT/work)
"""
from __future__ import annotations
import glob
import json
import os
import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2]


def _slash(p) -> str:
    p = os.fspath(p)
    return p if p.endswith(('/', '\\')) else p + os.sep


def root() -> Path:
    return Path(os.environ.get('BER_ROOT', os.getcwd())).resolve()


def ce_data() -> str:
    return _slash(os.environ.get('CE_DATA', root() / 'work' / 'ce'))


def ce_prep() -> str:
    bundled = _PKG / 'model' / 'prep'
    default = bundled if (bundled / 'prepared_gate0_full_dev_v1' / 'reference_ids.npy').exists() else root() / 'work' / 'prep'
    return _slash(os.environ.get('CE_PREP', default))


def owner_dir() -> Path:
    return Path(os.environ.get('BER_OWNER', _PKG / 'model' / 'owner_assignment'))


def weights_dir() -> Path:
    return Path(os.environ.get('BER_WEIGHTS', _PKG / 'model' / 'encoders'))


def dataset() -> str:
    if 'BER_DATA' in os.environ:
        return _slash(os.environ['BER_DATA'])
    for cand in (
        root() / 'data',
        root() / 'dataset',
    ):
        if cand.exists():
            return _slash(cand)
    return _slash(root() / 'data')


def store(split: str) -> str:
    parent = Path(os.environ.get('BER_STORE', root() / 'work'))
    return str(parent / f'{split}_store_v2' / 'records.sqlite3')


def v2_dir() -> str:
    return _slash(os.environ.get('BER_V2', root() / 'work' / 'v2'))


E5_SMALL_ID = 'intfloat/multilingual-e5-small'
E5_BASE_ID = 'intfloat/multilingual-e5-base'
# Pinned Hub commits for the E5 bases used at scoring time.
E5_SMALL_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
E5_BASE_REVISION = 'd128750597153bb5987e10b1c3493a34e5a4502a'


def e5_small() -> str:
    bundled = weights_dir() / 'e5_tokenizer'
    if bundled.exists() and 'BER_E5' not in os.environ:
        return str(bundled)
    return os.environ.get('BER_E5', E5_SMALL_ID)


def e5_base() -> str:
    return os.environ.get('BER_E5_BASE', E5_BASE_ID)


def e5_small_revision() -> str:
    return os.environ.get('BER_E5_REVISION', E5_SMALL_REVISION)


def e5_base_revision() -> str:
    return os.environ.get('BER_E5_BASE_REVISION', E5_BASE_REVISION)


def pretrained_kwargs(model: str | None = None) -> dict:
    """Hub loads use a pinned commit. Local snapshots already are that commit."""
    path = model if model is not None else e5_small()
    if os.path.isdir(path):
        return {}
    if path in (E5_SMALL_ID, os.environ.get('BER_E5', E5_SMALL_ID)):
        return {'revision': e5_small_revision()}
    if path in (E5_BASE_ID, os.environ.get('BER_E5_BASE', E5_BASE_ID)):
        return {'revision': e5_base_revision()}
    override = os.environ.get('BER_PRETRAINED_REVISION')
    return {'revision': override} if override else {}


def policy_json() -> Path:
    d = owner_dir()
    for p in (d / 'policy.json', d / 'owner_eval' / 'policy.json'):
        if p.exists():
            return p
    return d / 'policy.json'


def text_scorer_json() -> Path:
    d = owner_dir()
    for p in (d / 'text_scorer.json', d.parent / 'text_scorer.json' if d.name == 'owner_eval' else d / 'text_scorer.json'):
        if p.exists():
            return p
    return d / 'text_scorer.json'


def band_chunk_paths(pattern: str) -> list[str]:
    """Sort band_test_N paths by N (0,1,2,…), not lexicographically (0,1,10,11,2)."""
    def key(path):
        match = re.search(r'band_test_(\d+)', os.path.basename(path))
        return (int(match.group(1)) if match else 10**9, path)
    return sorted(glob.glob(pattern), key=key)


def _empty_route_complete(pattern: str) -> bool:
    marker = Path(pattern).parent / 'complete.json'
    if not marker.exists():
        return False
    meta = json.loads(marker.read_text())
    return bool(meta.get('complete')) and int(meta.get('rows', -1)) == 0


def expand_route_globs(spec: str, *, strict: bool = False) -> list[str]:
    """Expand `glob[;glob]`. Strict mode checks each pattern on its own.

    A pattern that matches nothing is allowed only when its directory has
    `complete.json` with `{"complete": true, "rows": 0}`.
    """
    files = []
    missing = []
    for pattern in spec.split(';'):
        pattern = pattern.strip()
        if not pattern:
            continue
        matched = sorted(glob.glob(pattern))
        if matched:
            files.extend(matched)
            continue
        if _empty_route_complete(pattern):
            continue
        missing.append(pattern)
    if strict and missing:
        raise FileNotFoundError(
            'empty extra route(s) without zero-row complete.json: ' + ';'.join(missing))
    return files


def _declared_partition_count(obj) -> int | None:
    if not isinstance(obj, dict):
        return None
    for key in ('partitions', 'n_partitions', 'expected_partitions'):
        value = obj.get(key)
        if value not in (None, ''):
            count = int(value)
            if count < 1:
                raise ValueError(f'{key} must be >= 1')
            return count
    return None


def _count_from_queue(queue_path: Path) -> int | None:
    data = json.loads(queue_path.read_text())
    count = _declared_partition_count(data)
    if count is None:
        progress = queue_path.parent / 'progress.json'
        if progress.exists():
            count = _declared_partition_count(json.loads(progress.read_text()))
    if count is None:
        for spec in (data.get('specs') or {}).values():
            argv = spec.get('argv') or []
            if '--partitions' in argv:
                count = int(argv[argv.index('--partitions') + 1])
                break
    return count


def _count_from_parts_glob(parts_glob: str) -> int | None:
    found = []
    for directory in glob.glob(parts_glob):
        for name in ('config.json', 'manifest.json'):
            path = Path(directory) / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            count = _declared_partition_count(payload)
            if count is None:
                count = _declared_partition_count(payload.get('config'))
            if count is not None:
                found.append(count)
                break
    if not found:
        return None
    if len(set(found)) != 1:
        raise ValueError(f'inconsistent declared partitions in BER_TEST_PARTS: {sorted(set(found))}')
    return found[0]


def expected_test_partitions() -> set[int]:
    """Partition ids a complete export must contain.

    Uses the declared total, never the set of files that already exist.
    Order: BER_TEST_PARTITIONS, queue/progress `partitions`, BER_TEST_PARTS
    config/manifest `partitions`, else the documented 128-way split.
    """
    env = os.environ.get('BER_TEST_PARTITIONS')
    if env not in (None, ''):
        count = int(env)
        if count < 1:
            raise ValueError('BER_TEST_PARTITIONS must be >= 1')
        return set(range(count))
    counts = []
    queue = os.environ.get('BER_QUEUE', '')
    default_queue = Path(v2_dir()) / 'inference' / 'queue.json'
    queue_path = Path(queue) if queue else (default_queue if default_queue.exists() else None)
    if queue_path and queue_path.exists():
        count = _count_from_queue(queue_path)
        if count is not None:
            counts.append(count)
    parts_glob = os.environ.get('BER_TEST_PARTS')
    if parts_glob:
        count = _count_from_parts_glob(parts_glob)
        if count is not None:
            counts.append(count)
    if counts:
        if len(set(counts)) != 1:
            raise ValueError(f'inconsistent declared partition totals: {sorted(set(counts))}')
        return set(range(counts[0]))
    return set(range(128))


def part_pkl_id(path: str) -> int:
    match = re.search(r'part_(\d+)\.pkl$', os.path.basename(path))
    if not match:
        raise ValueError(f'not a test part pickle: {path}')
    return int(match.group(1))
