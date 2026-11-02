"""Bounded official IndicXlit inference without hosted API/dictionary rescoring.

Loads the publisher's batch translator directly: avoids unrelated web-server,
Urdu-normalization and TensorFlow dependencies. This pilot covers Devanagari
names only, using both Hindi/Marathi hypotheses, not language identification.
"""
import argparse
from collections import Counter
import importlib.util
import json
import re
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
from v3_transfer_data import PILOT, load, write, digest

PATTERN = re.compile('[\u0900-\u0963\u0971-\u097f]+')  # preserve native digits and danda


def variants(smoke=False):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = PILOT / ('indic_smoke' if smoke else 'indic_features')
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    package = Path(sys.prefix) / 'lib/python3.10/site-packages/ai4bharat/transliteration/transformer'
    source = package / 'custom_interactive.py'
    spec = importlib.util.spec_from_file_location('official_indic_batch', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = PILOT / 'models/indic'
    archive = model / 'official-indic-en-v1.0.zip'
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = (model / info.filename).resolve()
            if not path.is_relative_to(model.resolve()):
                raise ValueError('Unsafe archive member')
            if not path.exists():
                z.extract(info, model)
    langs = package / 'models/indic2en/lang_list.txt'
    languages = [s for s in langs.read_text().splitlines() if s and s != 'en']
    translator = module.Transliterator(str(model / 'corpus-bin'), str(model / 'transformer/indicxlit.pt'),
        ','.join(f'{s}-en' for s in languages), str(langs), beam=4, batch_size=64)
    if smoke:
        result = translator.translate(['__hi__ ' + ' '.join(word) for word in ['दिल्ली', 'मुंबई']], nbest=2)
        print(result, flush=True)
        write(out / 'result.json', dict(complete=True, result=result, checkpoint_sha256=digest(model / 'transformer/indicxlit.pt')))
        return
    rows = [json.loads(line) for line in (PILOT / 'data/records.jsonl').open()]
    train = load('train')
    used = np.unique(np.r_[train['left'], train['right']])
    freq = Counter(token for i in used for token in PATTERN.findall(rows[int(i)]['name']) if len(token) <= 32)
    # Vocabulary selected on training inputs only. Unseen tokens retain the
    # existing AnyAscii spelling. This is a cheap lexical romanization pilot.
    tokens = [token for token, _ in sorted(freq.items(), key=lambda p: (-p[1], p[0]))[:15000]]
    del train
    cache = {}
    for language in ('hi', 'mr'):
        for j in range(0, len(tokens), 64):
            words = tokens[j:j+64]
            inputs = [f'__{language}__ ' + ' '.join(unicodedata.normalize('NFC', word).lower()) for word in words]
            with torch.inference_mode():
                result = translator.translate(inputs, nbest=2)
            outputs = [[] for _ in words]
            for line in result.splitlines():
                if line.startswith('H-'):
                    key, score, value = line.split('\t', 2)
                    outputs[int(key[2:])].append(value.replace(' ', ''))
            if any(not value for value in outputs):
                raise RuntimeError('Transliteration output alignment failure')
            for word, value in zip(words, outputs):
                cache.setdefault(word, {})[language] = value
            if j % 1024 == 0:
                write(out / 'progress.json', dict(stage='romanize', language=language, tokens=j+len(words),
                    tokens_per_language=len(tokens), seconds=time.monotonic()-start))
                print(json.dumps(dict(language=language, tokens=j+len(words), total=len(tokens))), flush=True)
    write(out / 'variants.json', cache)
    write(out / 'romanization_manifest.json', dict(complete=True, unique_training_tokens=len(tokens),
        languages=['hi', 'mr'], scope='Devanagari names, training-vocabulary selected, <=32 chars, 15k unique cap',
        beam=4, topk=2, dictionary_rescoring=False, license='MIT', seconds=time.monotonic()-start,
        source_sha256=digest(source), archive_sha256=digest(archive), checkpoint_sha256=digest(model / 'transformer/indicxlit.pt'),
        data_manifest_sha256=digest(PILOT / 'data/manifest.json')))


def features():
    from anyascii import anyascii
    from rapidfuzz.fuzz import ratio, token_sort_ratio
    out = PILOT / 'indic_features'
    ascii_out = PILOT / 'indic_ascii_features'
    ascii_out.mkdir(exist_ok=False)
    cache = json.loads((out / 'variants.json').read_text())
    rows = [json.loads(line) for line in (PILOT / 'data/records.jsonl').open()]

    def normalize(s):
        return ' '.join(re.sub(r'[^a-z0-9]+', ' ', anyascii(s).lower()).split())

    names, covered = [], []
    for row in rows:
        original = row['name']
        vals = [normalize(original)]
        tokens = PATTERN.findall(original)
        hits = sum(t in cache for t in tokens)
        if hits:
            for lang in ('hi', 'mr'):
                for rank in (0, 1):
                    def repl(match):
                        word = match.group()
                        values = cache.get(word, {}).get(lang)
                        return values[min(rank, len(values)-1)] if values else anyascii(word)
                    vals.append(normalize(PATTERN.sub(repl, original)))
        names.append(list(dict.fromkeys(vals)))
        covered.append(hits / len(tokens) if tokens else 0.)
    stats = {}
    for split in ('train', 'tune', 'dev'):
        a = load(split)
        matrix = np.zeros((len(a['y']), 4), dtype=np.float32)
        control = np.zeros_like(matrix)
        for i, (l, r) in enumerate(zip(a['left'], a['right'])):
            left, right = names[l], names[r]
            base = ratio(left[0], right[0])/100
            best = max(ratio(x, y)/100 for x in left for y in right)
            token = max(token_sort_ratio(x, y)/100 for x in left for y in right)
            matrix[i] = [best, token, best-base, max(covered[l], covered[r])]
            control[i] = [base, token_sort_ratio(left[0], right[0])/100, 0., max(covered[l], covered[r])]
        np.save(out / f'{split}_features.npy', matrix)
        np.save(ascii_out / f'{split}_features.npy', control)
        stats[split] = dict(pairs_with_coverage=int((matrix[:, 3] > 0).sum()),
            positives_with_coverage=int(((matrix[:, 3] > 0) & (a['y'] == 1)).sum()),
            improved_name_similarity=int((matrix[:, 2] > .01).sum()))
    write(out / 'manifest.json', dict(complete=True, splits=stats,
        data_manifest_sha256=digest(PILOT / 'data/manifest.json'),
        extra_features=['max_name_ratio', 'max_token_sort', 'gain_over_anyascii', 'covered_token_fraction']))
    write(ascii_out / 'manifest.json', dict(complete=True,
        data_manifest_sha256=digest(PILOT / 'data/manifest.json'),
        purpose='Same normalization/coverage/new feature count, without IndicXlit variants',
        extra_features=['anyascii_name_ratio', 'anyascii_token_sort', 'zero_gain', 'covered_token_fraction']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['variants', 'features', 'smoke'])
    args = parser.parse_args()
    if args.mode == 'features':
        features()
        # Finish the predeclared normalization-only comparator before the queue
        # runs the IndicXlit-feature model. No choice is made from dev scores.
        import gc
        gc.collect()
        from v3_transfer_cpu import fit
        fit('indic_ascii_control', extras='indic_ascii_features')
    else:
        variants(args.mode == 'smoke')
