"""Tune-only neural fusion and consolidated exploratory pilot results."""
import argparse
import json
import time
import numpy as np

from v3_transfer_data import PILOT, load, write, digest, metric, owner_scores, tune, bootstrap
from v3_transfer_cpu import compare


def evaluate_xlmr():
    out = PILOT / 'xlmr'
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['data_manifest_sha256'] == digest(PILOT / 'data/manifest.json')
    arrays = {s: load(s) for s in ('tune', 'dev')}
    probabilities = {s: np.load(out / f'{s}_probabilities.npy') for s in arrays}
    assert all(len(probabilities[s]) == len(arrays[s]['y']) and np.isfinite(probabilities[s]).all() for s in arrays)
    score, threshold = tune(arrays['tune'], probabilities['tune'])
    report = dict(complete=True, tag='xlmr', threshold=threshold, tune_macro_f05=score,
        dev=metric(arrays['dev'], probabilities['dev'], threshold), seconds=manifest['total_seconds'],
        data_manifest_sha256=manifest['data_manifest_sha256'])
    compare(arrays, probabilities, report)
    write(out / 'report.json', report)


def summarize():
    reports = {}
    for tag in ('control', 'full_negative_control', 'balanced', 'e5', 'importance_control', 'importance_e5', 'contrastive', 'indic_ascii_control', 'indic', 'xlmr'):
        path = PILOT / tag / 'report.json'
        if path.exists():
            reports[tag] = json.loads(path.read_text())
    if 'e5' in reports and 'contrastive' in reports:
        a = load('dev')
        reports['contrastive']['versus_frozen_e5'] = bootstrap(
            owner_scores(a, np.load(PILOT / 'e5/dev_probabilities.npy'), reports['e5']['threshold']),
            owner_scores(a, np.load(PILOT / 'contrastive/dev_probabilities.npy'), reports['contrastive']['threshold']))
    if 'indic' in reports and 'indic_ascii_control' in reports:
        a = load('dev')
        reports['indic']['versus_anyascii_feature_control'] = bootstrap(
            owner_scores(a, np.load(PILOT / 'indic_ascii_control/dev_probabilities.npy'), reports['indic_ascii_control']['threshold']),
            owner_scores(a, np.load(PILOT / 'indic/dev_probabilities.npy'), reports['indic']['threshold']))
    write(PILOT / 'summary.json', dict(complete=all(t in reports for t in ('control', 'balanced', 'e5', 'contrastive', 'indic_ascii_control', 'indic', 'xlmr')), reports=reports,
        caution='Exploratory reused development data; no French labels; no production changes; all five families include simplified/bounded approximations.',
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    lines = ['# Transfer pilot results', '', 'Exploratory local owner-macro F0.5; **not leaderboard scores**. See `TRANSFER_PILOT_PLAN.md` for controls and limitations.', '',
        '| Approach | Standalone dev | TP / FP | Matched-control delta [95% CI] | Tune-selected fusion with 200k |',
        '| --- | ---: | --- | --- | --- |']
    for tag, r in reports.items():
        delta = r['comparisons'].get('matched_control')
        ci = 'control' if delta is None else f"{delta['delta']:+.6f} [{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}]"
        fusion = r['fusion']['200000']
        lines.append(f"| {tag} | {r['dev']['macro_f05']:.6f} | {r['dev']['tp']} / {r['dev']['fp']} | {ci} | weight={fusion['weight']}, dev={fusion['dev']['macro_f05']:.6f} |")
    lines += ['', 'Saved 100k development baseline: 0.969362; 200k: 0.971523.', '',
        'A zero fusion weight means tuning chose to keep the production model unchanged. An interval spanning zero does not establish improvement. Full per-country/source/script/missing-address slices and paired 99% intervals are in the JSON reports.', '',
        'No recipe should be promoted solely because it beats the smaller sampled control. Confirm a promising recipe at larger matched training size and on genuinely unused owners before a full inference pass.', '']
    (PILOT / 'RESULTS.md').write_text('\n'.join(lines))
    from v3_transfer_diagnose import run as diagnose
    diagnose()
    from v3_transfer_routes import run as selective_fusion
    selective_fusion()
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evaluate-xlmr', action='store_true')
    args = p.parse_args()
    evaluate_xlmr() if args.evaluate_xlmr else summarize()
