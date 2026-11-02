"""Reserve a new eval-owner sample without reading labels or running a model.

Exclude the recorded historic selections, the already-scored 10k audit,
current tune/dev, and the original neural rival owners. This is a reservation,
not a claim that every other session's historical inspection is knowable.
Do not use these owners for fitting or choose a model from their future score.
"""
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path

from ber.data import FIELDS, read_tsv, write_tsv
from ber.store import connect
from v2_prepare_inputs import reservoir
from v3_transfer_data import ROOT, digest, write


def validate_selection(rows, excluded, expected):
    ids = [r[0] for r in rows]
    if len(ids) != expected or len(set(ids)) != expected or set(ids) & excluded:
        raise ValueError('Fresh audit size, uniqueness or exclusion failed')


def run():
    output = ROOT/'artifacts/improvement_v3/reserved_fresh_audit2k_v1'
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'selection.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if digest(output/'references.tsv') != old['references_sha256']:
            raise ValueError('Reserved audit reference identity changed')
        print('Existing reserved selection verified; no labels read.')
        return
    catalog_path = ROOT/'artifacts/improvement_v2/selection/inputs_v1/manifest.json'
    catalog = json.loads(catalog_path.read_text())
    paths = {Path(p):v['sha256'] for p,v in catalog['historic_reference_sources'].items()}
    paths.update({Path(v['path']):v['sha256'] for v in catalog['sets'].values()})
    excluded = set(); sources = {}
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError('Recorded historic selection changed: '+str(path))
        ids = {r['entity_id'] for r in read_tsv(path)}
        excluded.update(ids); sources[str(path)] = dict(owners=len(ids), sha256=expected)
    # Auxiliary owner texts have already been examined. Ignore every label field.
    rival_path = Path('work/ce/rivals.pkl')
    rows = pickle.load(rival_path.open('rb'))
    rival_ids = {r[0] for r in rows}; excluded.update(rival_ids); del rows
    sources[str(rival_path)] = dict(owners=len(rival_ids), sha256=digest(rival_path), fields_used=['owner_id'])
    db = connect(ROOT/'artifacts/train_store_v2'); db.execute('PRAGMA cache_size=-8192')
    rows, eligible = reservoir(db.execute(
        "SELECT entity_id,name,address,country FROM anchors WHERE fold='eval' ORDER BY entity_id"),
        2000, 202609269, excluded)
    db.close(); validate_selection(rows, excluded, 2000)
    write_tsv(output/'references.tsv', (dict(zip(FIELDS,r)) for r in rows), FIELDS)
    write(manifest, dict(complete=True, owners=2000, fold='eval', seed=202609269,
        country_counts=dict(Counter(r[3] for r in rows)), eligible_owners=eligible,
        excluded_owners=len(excluded), exclusions_sha256=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
        source_catalog_sha256=digest(catalog_path), sources=sources,
        references_sha256=digest(output/'references.tsv'), code_sha256=digest(__file__),
        selection_reads_ownership_labels=False, labels_scored=False, training_allowed=False,
        policy='Freeze complete model/threshold/ownership recipe before a single reporting-only evaluation. No fitting, tuning or runner-up selection on these owners.',
        limitations=['Excludes known recorded selection and neural-rival owners, not proof of all independent-session activity.',
            'Provided gallery text is shared with retrieval; this is not an unseen-gallery or French evaluation.',
            'Country mix remains US/India only; public score cannot be inferred directly.']))
    print('Reserved 2,000 eval owners; no candidate generation, label lookup or scoring.')


if __name__ == '__main__':
    run()
