"""Inspect peer dictionary evidence and held-out native-name failures read-only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import unicodedata

sys.path.insert(0, 'src/neural')
from name_unique_lib import core

ROOT = Path(__file__).resolve().parents[1]
C = Path('work/ce')
OUT = ROOT/'artifacts/improvement_v3/keychannel_dictionary_audit_v1'
NAT = re.compile(r'[\u0900-\u0DFF]')


def toks(s):
    return [t for t in re.split(r'[\s,()\[\]/&]+', s) if t]


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [C/'translit_cnt.pkl', C/'err_links.pkl', C/'keych_devtune.pkl', C.parent/'key_channels.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cnt = pickle.load((C/'translit_cnt.pkl').open('rb'))
    d = {p: c.most_common(1)[0][0] for p,c in cnt.items()
         if c.most_common(1)[0][1] >= 2 and c.most_common(1)[0][1]/sum(c.values()) >= .4}
    norm = lambda s: unicodedata.normalize('NFC',s).strip(' .:;!?\"\u201c\u201d\u2018\u2019-')
    nd = {norm(k): v for k,v in d.items()}
    rows = pickle.load((C/'err_links.pkl').open('rb'))
    found = {(a,t) for a,t,ch in pickle.load((C/'keych_devtune.pkl').open('rb')) if ch == 'translit'}
    stats = defaultdict(Counter); unknown = Counter(); unknown_miss = Counter(); failures = []
    for r in rows:
        if not r['y'] or not r['native']:
            continue
        original = r['t_text'][0]; owner = r['s1_text'][0]
        tt = toks(original); nt = [t for t in tt if NAT.search(t)]; missing = [t for t in nt if t not in d]
        translated = ' '.join(d[t] if t in d else t for t in tt if not NAT.search(t) or t in d)
        normalized = ' '.join(nd.get(norm(t), t) for t in tt)
        match = core(translated) == core(owner)
        nmatch = core(normalized) == core(owner)
        pair = (r['s1'],r['t']); present = pair in found
        for key in (r['sp']+'/all_native_true',r['sp']+'/'+r['stage']):
            c = stats[key]; c['pairs'] += 1; c['unknown_token_pairs'] += bool(missing)
            c['all_tokens_mapped'] += not missing; c['mapped_core_matches_owner'] += match
            c['normalized_mapped_core_matches_owner'] += nmatch
            c['present_in_translit_candidate_channel'] += present
            c['matching_key_but_channel_dropped'] += match and not present
        unknown.update(missing)
        if r['stage'] != 'accepted':
            unknown_miss.update(missing)
            failures.append(dict(split=r['sp'], owner=r['s1'], target=r['t'], stage=r['stage'],
                owner_name=owner,target_name=original,owner_address=r['s1_text'][1],target_address=r['t_text'][1],
                unknown_tokens=missing, translated=translated,mapped_key=core(translated),
                owner_key=core(owner),key_equal=match,normalization_key_equal=nmatch,
                in_translit_candidates=present))
    reps = {w:[dict(native=p, count=int(c[w]), selected=d.get(p)) for p,c in cnt.items() if w in c]
            for w in ('sky','apex','finance')}
    report = dict(raw_native_entries=len(cnt),selected_native_entries=len(d),
        selected_distinct_english_words=len(set(d.values())),total_alignment_votes=sum(sum(c.values()) for c in cnt.values()),
        keys_with_multiple_english_options=sum(len(c)>1 for c in cnt.values()),
        rejected_support=sum(max(c.values())<2 for c in cnt.values()),
        rejected_dominance=sum(max(c.values())/sum(c.values())<.4 for c in cnt.values()),
        queried_word_examples=reps,slices={k:dict(v) for k,v in stats.items()},
        most_common_unknown_true_tokens=unknown.most_common(30),
        unknown_tokens_in_remaining_errors=unknown_miss.most_common(30),
        caveats=['Counts inspect the saved alignment cache, not an independent re-scan of the training builder.',
            'err_links.pkl uses the provisional k10 highest-probability ownership diagnostic, not the frozen deployed text-margin policy.',
            'Cases are diagnostic only; no held-out labels used to change a dictionary or submission.',
            'Dictionary size counts unique token types; native-name record count counts occurrences.'],inputs_sha256=hashes)
    for p,h in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (OUT/'failures.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    run()
