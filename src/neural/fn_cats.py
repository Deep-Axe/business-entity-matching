import json, re, collections
from rapidfuzz import fuzz

rows = [json.loads(l) for l in open('artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl')]
num = re.compile(r'\d+')

def nums(s):
    return num.findall(s or '')

def house_rel(a, b):
    na, nb = nums(a), nums(b)
    if not na or not nb:
        return 'no_numbers'
    x = na[0]
    if x in nb:
        return 'equal'
    for y in nb:
        if len(y) == len(x) - 1 and (x[1:] == y or x[:-1] == y):
            return 'digit_deleted'
        if len(y) == len(x) and sum(c != d for c, d in zip(x, y)) == 1:
            return 'one_digit_changed' if abs(int(x) - int(y)) >= 10 or len(x) == 1 else 'last_digit_changed'
    for y in nb:
        if abs(int(x) - int(y)) <= 25:
            return 'small_offset'
    return 'different'

def cat(r):
    a, b = r['reference'], r['target']
    an, bn = a['business_name'], b['business_name']
    ta = b['business_address'].strip()
    if not ta or ta.upper() in ('NULL', '<NULL>'):
        return 'target_address_empty'
    if any(ord(c) > 0x900 for c in bn):
        return 'native_script_name'
    if fuzz.token_set_ratio(an.lower(), bn.lower()) < 50:
        return 'unrelated_name(pseudo/dba)'
    return 'house:' + house_rel(a['business_address'], ta)

if __name__ == '__main__':
    for kind in ['retrieved_false_negative', 'retrieval_miss', 'false_positive']:
        sel = [r for r in rows if r['kind'] == kind]
        c = collections.Counter(cat(r) for r in sel)
        print(kind, len(sel))
        for k, v in c.most_common():
            print(f'   {v:4d}  {k}')
