import json, sqlite3
con = sqlite3.connect('file:artifacts/train_store_v2/records.sqlite3?mode=ro', uri=True)
rows = [json.loads(l) for l in open('artifacts/improvement_v2/research_next/dev100k_error_analysis_v1/error_pairs.jsonl')]
fp = [r for r in rows if r['kind'] == 'false_positive']
for r in fp:
    o = con.execute('select reference_id from labels where target_id=?', (r['target_id'],)).fetchone()
    if not o:
        continue
    rec = con.execute('select name,address from anchors where entity_id=?', (o[0],)).fetchone()
    a, b = r['reference'], r['target']
    print(f"{r['probability']:.2f} S1 : {a['business_name']} | {a['business_address']}\n     T  : {b['business_name']} | {b['business_address']}\n     OWN: {rec[0]} | {rec[1]}")
