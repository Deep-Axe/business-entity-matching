"""Explain complementarity/failure slices versus frozen 200k, not score ranking."""
import json
import numpy as np
from v3_transfer_data import PILOT, load, write


def run():
    a = load('dev')
    y = a['y'] == 1
    baseline = a['baseline_200000'] >= .70
    slices = {'all': np.ones(len(y), dtype=bool),
        'India': a['countries'][a['query_ids']] == 'India',
        'US': a['countries'][a['query_ids']] == 'US',
        'cross_script': a['cross_script'], 'missing_address': a['missing_address'],
        'both_addresses': ~a['missing_address'],
        'source2': a['target_source'] == 2, 'source3': a['target_source'] == 3}
    result = dict(complete=True, purpose='Diagnostic only; no policy selected from dev. Slices overlap and must not be summed.', models={})
    for tag in ('control', 'full_negative_control', 'balanced', 'e5', 'importance_control', 'importance_e5',
                'contrastive', 'indic_ascii_control', 'indic', 'xlmr'):
        path = PILOT / tag / 'report.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        p = np.load(PILOT / tag / 'dev_probabilities.npy') >= r['threshold']
        result['models'][tag] = dict(macro_f05=r['dev']['macro_f05'], threshold=r['threshold'],
            slices={name: dict(recovered_true_matches=int((m & p & ~baseline & y).sum()),
                              introduced_false_matches=int((m & p & ~baseline & ~y).sum()),
                              lost_true_matches=int((m & ~p & baseline & y).sum()),
                              removed_false_matches=int((m & ~p & baseline & ~y).sum()))
                    for name, m in slices.items()})
    write(PILOT / 'complementarity.json', result)
    return result


if __name__ == '__main__':
    run()
